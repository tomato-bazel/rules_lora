#!/usr/bin/env python3
"""Local LoRA training entry point — `lora_train` on the `local` backend.

A `py_binary`, not a shell script: the //lora:defs.bzl macro emits a
`<name>_local` py_binary with this as `srcs`, the build-generated
`<name>.local.json` config + the dataset in runfiles, and the config/dataset
runfiles paths passed via `args` (`$(rlocationpath ...)`). Replaces the former
`local_runner.sh` + its generated bash wrapper, so rules_lora no longer emits
or execs any shell of its own.

This stays a thin *orchestrator*: torch / torchtune live in a runtime venv
(`.venvs/lora-local`) it creates + drives via subprocess — it does not import
them, so the py_binary itself has no heavy deps and stays hermetic to analyze.
The training run (venv, HF download, `tune run`) is inherently non-hermetic
(network + the host accelerator); hermetic vendoring of torch is a follow-up.

Config schema (`<name>.local.json`, written by the lora_local_config rule):
    {
      "name": str, "base_id": str, "base_revision": str, "family": str,
      "rank": int, "alpha": int, "target_modules": [str, ...],
      "learning_rate": str, "micro_batch_size": int,
      "grad_accum_steps": int, "epochs": int,
      "dataset": str  # runfiles-relative path of the validated JSONL
    }
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from python.runfiles import runfiles

# family -> (tokenizer component, HF checkpointer model_type). The LoRA *model*
# builder is NOT here: it depends on the base model's parameter size, not the
# family (qwen2 alone ships 0.5B/1.5B/3B/7B builders). Picking the wrong size
# fails at checkpoint load with a "size mismatch" (e.g. norm.scale 896 vs 1536
# for 0.5B vs 1.5B). See _model_builder.
_FAMILIES = {
    "qwen2": ("torchtune.models.qwen2.qwen2_tokenizer", "QWEN2"),
    "llama3": ("torchtune.models.llama3.llama3_tokenizer", "LLAMA3"),
    "mistral": ("torchtune.models.mistral.mistral_tokenizer", "MISTRAL"),
}


def _model_builder(family, base_id):
    """The torchtune LoRA model builder matching the base model's size.

    Derives the size from the HF id (e.g. Qwen2.5-0.5B-Instruct -> "0.5") and
    maps to the size-suffixed builder (torchtune.models.qwen2.lora_qwen2_0_5b).
    Must match the checkpoint's hidden size or `tune run` errors at load.
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", base_id)
    size = m.group(1).replace(".", "_") if m else None
    if family == "qwen2":
        if not size:
            raise SystemExit("cannot parse model size from base_id: %s" % base_id)
        return "torchtune.models.qwen2.lora_qwen2_%sb" % size
    if family == "llama3":
        return "torchtune.models.llama3.lora_llama3_%sb" % size if size \
            else "torchtune.models.llama3.lora_llama3_8b"
    if family == "mistral":
        return "torchtune.models.mistral.lora_mistral"  # torchtune ships the 7B builder
    raise SystemExit("unknown family: %s" % family)

# Known-good pin set for the Apple-Silicon-MPS path. The torchtune + torchao +
# kagglehub triangle breaks in interesting ways on every unpinned release; this
# set is the one observed working (see the former local_runner.sh notes):
#   torchtune 0.4.0 has lora_qwen2_1_5b and skips int4_weight_only during MPS;
#   torchao 0.5.0 still exports int4_weight_only (the name torchtune references);
#   kagglehub<0.3 avoids the kagglesdk get_web_endpoint import break.
_PIP_PINS = [
    "torch",
    "torchao==0.5.0",
    "torchtune==0.4.0",
    "kagglehub<0.3",
    "huggingface_hub[cli]",
    "transformers",
    "datasets",
]


def _log(name, msg):
    print("[lora-%s] local: %s" % (name, msg), file=sys.stderr, flush=True)


def _detect_device(venv_python, name):
    """cpu | mps | cuda, probed post-install so `import torch` resolves."""
    if sys.platform == "darwin":
        probe = subprocess.run(
            [venv_python, "-c", "import torch; assert torch.backends.mps.is_available()"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def _render_config(cfg, model_dir, output_dir, dataset, device):
    tokenizer, model_type = _FAMILIES[cfg["family"]]
    model_lora = _model_builder(cfg["family"], cfg["base_id"])
    modules = ", ".join('"%s"' % m for m in cfg["target_modules"])
    return """# Rendered by rules_lora local_train.py at run time.
output_dir: {output_dir}

tokenizer:
  _component_: {tokenizer}
  path: {model_dir}/vocab.json
  merges_file: {model_dir}/merges.txt
  max_seq_len: 2048

model:
  _component_: {model_lora}
  lora_attn_modules: [{modules}]
  apply_lora_to_mlp: False
  lora_rank: {rank}
  lora_alpha: {alpha}
  lora_dropout: 0.0

checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: {model_dir}
  checkpoint_files:
    - model.safetensors
  output_dir: {output_dir}
  model_type: {model_type}

dataset:
  _component_: torchtune.datasets.chat_dataset
  source: json
  data_files: {dataset}
  conversation_column: messages
  conversation_style: openai
  packed: false
  train_on_input: false

seed: 0
shuffle: True
batch_size: {micro_batch_size}
gradient_accumulation_steps: {grad_accum_steps}
epochs: {epochs}
max_steps_per_epoch: null
resume_from_checkpoint: False
save_adapter_weights_only: True

optimizer:
  _component_: torch.optim.AdamW
  weight_decay: 0.01
  lr: {learning_rate}
  fused: False

lr_scheduler:
  _component_: torchtune.modules.get_cosine_schedule_with_warmup
  num_warmup_steps: 1

loss:
  _component_: torchtune.modules.loss.CEWithChunkedOutputLoss

device: {device}
dtype: bf16

compile: False
enable_activation_checkpointing: False
metric_logger:
  _component_: torchtune.training.metric_logging.StdoutLogger
log_every_n_steps: 1
log_peak_memory_stats: True
profiler:
  _component_: torchtune.training.setup_torch_profiler
  enabled: False
""".format(
        output_dir=output_dir,
        tokenizer=tokenizer,
        model_dir=model_dir,
        model_lora=model_lora,
        modules=modules,
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        model_type=model_type,
        dataset=dataset,
        micro_batch_size=cfg["micro_batch_size"],
        grad_accum_steps=cfg["grad_accum_steps"],
        epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        device=device,
    )


def main():
    ap = argparse.ArgumentParser(description="Local LoRA training runner.")
    ap.add_argument("--config", required=True, help="rlocationpath of <name>.local.json")
    args = ap.parse_args()

    r = runfiles.Create()
    config_path = r.Rlocation(args.config)
    if not config_path or not os.path.isfile(config_path):
        sys.exit("fatal: config not found via runfiles: %s" % args.config)

    with open(config_path) as fh:
        cfg = json.load(fh)
    name = cfg["name"]
    if cfg["family"] not in _FAMILIES:
        sys.exit("fatal: unknown family %s" % cfg["family"])

    dataset_path = r.Rlocation(cfg["dataset"])
    if not dataset_path or not os.path.isfile(dataset_path):
        sys.exit("fatal: dataset not found via runfiles: %s" % cfg["dataset"])

    # Resolve the dataset to an absolute path before we cd into the workspace.
    dataset_abs = os.path.abspath(dataset_path)
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    os.chdir(workspace)
    _log(name, "cwd=%s adapter=%s family=%s" % (os.getcwd(), name, cfg["family"]))

    # ── venv (torch/torchtune live here; this binary only drives it) ──
    python_bin = (
        shutil.which("python3.11") or shutil.which("python3.12") or shutil.which("python3")
    )
    if not python_bin:
        sys.exit("fatal: no python3 on PATH")
    venv = os.path.join(workspace, ".venvs", "lora-local")
    if not os.path.isdir(venv):
        _log(name, "creating venv at %s" % venv)
        subprocess.run([python_bin, "-m", "venv", venv], check=True)
    venv_python = os.path.join(venv, "bin", "python")
    venv_tune = os.path.join(venv, "bin", "tune")
    venv_hf = os.path.join(venv, "bin", "hf")

    _log(name, "ensuring torch + torchao + torchtune")
    subprocess.run(
        [venv_python, "-m", "pip", "install", "--quiet"] + _PIP_PINS,
        check=True,
        stdout=sys.stderr.fileno(),
    )

    device = _detect_device(venv_python, name)
    _log(name, "device=%s" % device)

    # ── base model fetch ──
    _log(name, "pre-fetching %s@%s" % (cfg["base_id"], cfg["base_revision"]))
    dl = subprocess.run(
        [venv_hf, "download", "--revision", cfg["base_revision"], "--quiet", cfg["base_id"]],
        capture_output=True,
        text=True,
        check=True,
    )
    model_dir = dl.stdout.strip().splitlines()[-1]
    _log(name, "model staged at %s" % model_dir)

    # ── render config + train ──
    output_dir = os.path.join(os.getcwd(), "outputs", "adapter-%s" % name)
    os.makedirs(output_dir, exist_ok=True)
    rendered = _render_config(cfg, model_dir, output_dir, dataset_abs, device)
    with tempfile.NamedTemporaryFile(
        "w", suffix="-%s-local.yaml" % name, delete=False
    ) as fh:
        fh.write(rendered)
        config_yaml = fh.name

    _log(name, "invoking tune run on %s" % device)
    subprocess.run(
        [venv_tune, "run", "lora_finetune_single_device", "--config", config_yaml],
        check=True,
    )
    _log(name, "complete; outputs at %s" % output_dir)


if __name__ == "__main__":
    main()
