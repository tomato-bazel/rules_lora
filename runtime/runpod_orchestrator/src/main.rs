//! runpod_orchestrator — build-time tooling for the runpod backend
//! of `lora_train`.
//!
//! Two subcommands:
//!   * `write-jobspec`         — serialize the rule's attrs into a
//!                               TrainingJobSpec JSON consumed by
//!                               downstream targets.
//!   * `write-runpod-manifest` — synthesize the runpod-cli manifest
//!                               TOML whose `setup` + `run` blocks
//!                               actually invoke torchtune to train
//!                               the LoRA adapter on the pod.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "runpod_orchestrator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Build-time: serialize the rule attrs into a TrainingJobSpec
    /// JSON file. The execute-time `run` subcommand reads this.
    WriteJobspec {
        #[arg(long)]
        name: String,
        #[arg(long)]
        recipe: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        base_id: String,
        #[arg(long)]
        base_revision: String,
        #[arg(long)]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build-time: synthesize the runpod-cli manifest TOML whose
    /// `setup` and `run` blocks pull HF weights, render an effective
    /// torchtune config from the recipe attrs, and invoke
    /// `tune run lora_finetune_single_device` on the pod.
    WriteRunpodManifest {
        /// Job name (e.g., `parser_jobspec`).
        #[arg(long)]
        name: String,
        /// RunPod GPU type (e.g., `NVIDIA H100 80GB HBM3`).
        #[arg(long)]
        gpu_type: String,
        /// RunPod container image.
        #[arg(long)]
        image: String,
        /// HF hub repo id (e.g., `Qwen/Qwen2.5-1.5B-Instruct`).
        #[arg(long)]
        base_id: String,
        /// HF hub revision (commit sha or branch).
        #[arg(long)]
        base_revision: String,
        /// torchtune family (e.g., `qwen2`, `llama3`); selects the
        /// matching tokenizer + model_type component in the rendered
        /// config.
        #[arg(long, default_value = "qwen2")]
        family: String,
        /// LoRA rank.
        #[arg(long)]
        rank: u32,
        /// LoRA alpha.
        #[arg(long)]
        alpha: u32,
        /// Comma-separated target modules.
        #[arg(long)]
        target_modules: String,
        /// Learning rate (rendered into the recipe).
        #[arg(long)]
        learning_rate: String,
        /// Micro batch size.
        #[arg(long)]
        micro_batch_size: u32,
        /// Gradient accumulation steps.
        #[arg(long)]
        grad_accum_steps: u32,
        /// Number of epochs.
        #[arg(long)]
        epochs: u32,
        /// Destination path for the manifest TOML.
        #[arg(long)]
        out: PathBuf,
    },
    /// Execute-time: upload spec + dataset, poll the job, download
    /// the adapter. Not yet implemented; v0.1.
    Run {
        #[arg(long)]
        jobspec: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::WriteJobspec {
            name,
            recipe,
            dataset,
            base_id,
            base_revision,
            backend,
            out,
        } => write_jobspec(name, recipe, dataset, base_id, base_revision, backend, out),
        Cmd::WriteRunpodManifest {
            name,
            gpu_type,
            image,
            base_id,
            base_revision,
            family,
            rank,
            alpha,
            target_modules,
            learning_rate,
            micro_batch_size,
            grad_accum_steps,
            epochs,
            out,
        } => write_runpod_manifest(WriteRunpodManifestArgs {
            name,
            gpu_type,
            image,
            base_id,
            base_revision,
            family,
            rank,
            alpha,
            target_modules,
            learning_rate,
            micro_batch_size,
            grad_accum_steps,
            epochs,
            out,
        }),
        Cmd::Run { jobspec } => {
            anyhow::bail!(
                "runpod_orchestrator run: not implemented yet (v0.1). \
                 v0.0 only emits the spec at {}.",
                jobspec.display()
            )
        }
    }
}

fn write_jobspec(
    name: String,
    recipe: PathBuf,
    dataset: PathBuf,
    base_id: String,
    base_revision: String,
    backend: String,
    out: PathBuf,
) -> Result<()> {
    let recipe_bytes =
        std::fs::read(&recipe).with_context(|| format!("reading {}", recipe.display()))?;
    let dataset_bytes =
        std::fs::read(&dataset).with_context(|| format!("reading {}", dataset.display()))?;
    let recipe_sha = blake3::hash(&recipe_bytes).to_hex().to_string();
    let dataset_sha = blake3::hash(&dataset_bytes).to_hex().to_string();
    let recipe_yaml = String::from_utf8(recipe_bytes).context("recipe is not utf-8")?;

    let spec = serde_json::json!({
        "name": name,
        "recipe_sha": recipe_sha,
        "dataset_sha": dataset_sha,
        "base_model_id": base_id,
        "base_model_revision": base_revision,
        "backend": backend,
        "recipe_yaml": recipe_yaml,
        "backend_config_json": "{}",
        "max_minutes": 0,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&spec)?)
        .with_context(|| format!("writing {}", out.display()))?;
    eprintln!(
        "runpod_orchestrator: jobspec → {} (recipe={}…, dataset={}…)",
        out.display(),
        &recipe_sha[..12],
        &dataset_sha[..12]
    );
    Ok(())
}

// ─── write-runpod-manifest ──────────────────────────────────────────
//
// Synthesizes the runpod-cli manifest TOML. The `setup` and `run`
// blocks are real torchtune-invoking bash; placeholders (job name,
// HF model id, LoRA hyperparams) are interpolated at build time, the
// rest is static.
//
// Family → torchtune component mapping. v0 covers Qwen2 / Llama3 /
// Mistral; extension is a `match` arm + a tokenizer.json convention
// the family already uses upstream.

struct WriteRunpodManifestArgs {
    name: String,
    gpu_type: String,
    image: String,
    base_id: String,
    base_revision: String,
    family: String,
    rank: u32,
    alpha: u32,
    target_modules: String,
    learning_rate: String,
    micro_batch_size: u32,
    grad_accum_steps: u32,
    epochs: u32,
    out: PathBuf,
}

fn write_runpod_manifest(a: WriteRunpodManifestArgs) -> Result<()> {
    let target_modules_yaml = a
        .target_modules
        .split(',')
        .map(|s| format!("\"{}\"", s.trim()))
        .collect::<Vec<_>>()
        .join(", ");
    let family_components = match a.family.as_str() {
        "qwen2" => FamilyComponents {
            tokenizer: "torchtune.models.qwen2.qwen2_tokenizer",
            model_lora: "torchtune.models.qwen2.lora_qwen2",
            checkpoint_model_type: "QWEN2",
        },
        "llama3" => FamilyComponents {
            tokenizer: "torchtune.models.llama3.llama3_tokenizer",
            model_lora: "torchtune.models.llama3.lora_llama3",
            checkpoint_model_type: "LLAMA3",
        },
        "mistral" => FamilyComponents {
            tokenizer: "torchtune.models.mistral.mistral_tokenizer",
            model_lora: "torchtune.models.mistral.lora_mistral",
            checkpoint_model_type: "MISTRAL",
        },
        other => anyhow::bail!(
            "unknown model family '{}'. Supported: qwen2 / llama3 / mistral.",
            other
        ),
    };

    let toml = render_manifest_toml(&a, &target_modules_yaml, &family_components);
    std::fs::write(&a.out, toml)
        .with_context(|| format!("writing {}", a.out.display()))?;
    eprintln!(
        "runpod_orchestrator: manifest → {} ({} adapter, family={})",
        a.out.display(),
        a.name,
        a.family
    );
    Ok(())
}

struct FamilyComponents {
    tokenizer: &'static str,
    model_lora: &'static str,
    checkpoint_model_type: &'static str,
}

fn render_manifest_toml(
    a: &WriteRunpodManifestArgs,
    target_modules_yaml: &str,
    fc: &FamilyComponents,
) -> String {
    // Top-level keys (name, setup, run) come *before* the `[resources]`
    // table — otherwise TOML parses them as members of that table and
    // runpod-cli's Manifest struct rejects the manifest with
    // `missing field setup`.
    format!(
        r#"name = "lora-{name}"
workdir = "."
outputs = ["adapter-{name}"]

setup = """
set -euo pipefail
echo "[lora-{name}] setup: installing torchtune + huggingface-cli"

# RunPod's pytorch image ships python3 + pip + CUDA; we just add
# torchtune and the HF hub client.
pip install --quiet --no-input \
    torchtune \
    "huggingface_hub[cli]" \
    transformers \
    datasets

# Pre-fetch the base model. `hf download` is idempotent and prints
# the cached path on stdout — capture it for the train step.
echo "[lora-{name}] setup: pre-fetching {base_id}@{base_revision}"
hf download --revision {base_revision} --quiet {base_id} > /tmp/lora-{name}.model_dir
echo "[lora-{name}] setup: model staged at $(cat /tmp/lora-{name}.model_dir)"
"""

run = """
set -euo pipefail
echo "[lora-{name}] train: starting"

MODEL_DIR="$(cat /tmp/lora-{name}.model_dir)"
DATASET="$(find . -name '*.jsonl' -path '*lora_dataset*' -o -name 'dataset.jsonl' | head -1)"
if [[ -z "${{DATASET}}" ]]; then
    echo "[lora-{name}] train: ERROR — no dataset JSONL found in workdir" >&2
    exit 2
fi
OUTPUT_DIR="$(pwd)/outputs/adapter-{name}"
mkdir -p "${{OUTPUT_DIR}}"

# Synthesize a complete torchtune config by injecting the per-job
# paths into a static template. Keeping the recipe attrs at the top
# makes the rendered file diff-friendly when hyperparams change.
cat > /tmp/lora-{name}.yaml <<YAML
# Rendered by runpod_orchestrator write-runpod-manifest at build time.
# Hand-edits get overwritten next bazel run.

output_dir: ${{OUTPUT_DIR}}

tokenizer:
  _component_: {tokenizer}
  path: ${{MODEL_DIR}}/vocab.json
  merges_file: ${{MODEL_DIR}}/merges.txt
  max_seq_len: 2048

model:
  _component_: {model_lora}
  lora_attn_modules: [{target_modules_yaml}]
  apply_lora_to_mlp: False
  apply_lora_to_output: False
  lora_rank: {rank}
  lora_alpha: {alpha}

checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: ${{MODEL_DIR}}
  checkpoint_files:
    - model.safetensors
  output_dir: ${{OUTPUT_DIR}}
  model_type: {ckpt_type}

dataset:
  _component_: torchtune.datasets.chat_dataset
  source: json
  data_files: ${{DATASET}}
  conversation_column: messages
  conversation_style: openai
  packed: false
  train_on_input: false

seed: 0
shuffle: True
batch_size: {micro_batch_size}
gradient_accumulation_steps: {grad_accum_steps}
epochs: {epochs}

optimizer:
  _component_: torch.optim.AdamW
  weight_decay: 0.01
  lr: {learning_rate}

loss:
  _component_: torchtune.modules.loss.CEWithChunkedOutputLoss

device: cuda
dtype: bf16

compile: False
metric_logger:
  _component_: torchtune.training.metric_logging.StdoutLogger
log_every_n_steps: 1
log_peak_memory_stats: True
YAML

echo "[lora-{name}] train: invoking tune run"
tune run lora_finetune_single_device --config /tmp/lora-{name}.yaml
echo "[lora-{name}] train: complete; outputs at ${{OUTPUT_DIR}}"
"""

[resources]
gpu_type = "{gpu_type}"
image = "{image}"
"#,
        name = a.name,
        gpu_type = a.gpu_type,
        image = a.image,
        base_id = a.base_id,
        base_revision = a.base_revision,
        tokenizer = fc.tokenizer,
        model_lora = fc.model_lora,
        ckpt_type = fc.checkpoint_model_type,
        target_modules_yaml = target_modules_yaml,
        rank = a.rank,
        alpha = a.alpha,
        learning_rate = a.learning_rate,
        micro_batch_size = a.micro_batch_size,
        grad_accum_steps = a.grad_accum_steps,
        epochs = a.epochs,
    )
}
