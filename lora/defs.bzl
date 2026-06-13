"""rules_lora public API.

```starlark
load("@rules_lora//lora:defs.bzl",
     "lora_base_model",
     "lora_dataset",
     "lora_recipe",
     "lora_train",
     "lora_merge",
     "expert_manifest")
```

* `lora_dataset`, `lora_recipe`, `lora_base_model` — thin
  re-exports of the underlying rules.
* `lora_train` — v0.0.2: now a macro. Always emits the typed
  jobspec; additionally composes with `@rules_runpod` when
  `backend = "runpod"` to emit a synthesized manifest +
  `runpod_job`, giving the user `bazel run :<name>.runpod_job.run`.
* `lora_merge` — v0.0.34: fold a trained adapter into its base and
  export a standalone HF model dir (`bazel run :<name>.run`), with an
  optional HF push. Carries `LoraBaseModelInfo` so the merged model is
  usable directly as a `lora_train(base = ...)` (two-stage rebase).
* `expert_manifest` — bundle N adapters as the routing input.

Macros forward to rules in `//lora/private:rules.bzl`, which fill
providers from `//lora/private:providers.bzl`.
"""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_runpod//runpod:defs.bzl", "runpod_job", "runpod_manifest")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load(
    "//lora/private:aspects.bzl",
    _lora_lineage = "lora_lineage",
    _lora_lineage_aspect = "lora_lineage_aspect",
)
load(
    "//lora/private:rules.bzl",
    _lora_base_model = "lora_base_model",
    _lora_corpus = "lora_corpus",
    _lora_dataset = "lora_dataset",
    _lora_local_runner_rule = "lora_local_runner",
    _lora_merge_rule = "lora_merge",
    _lora_recipe = "lora_recipe",
    _lora_runpod_manifest_synth = "lora_runpod_manifest_synth",
    _lora_train_rule = "lora_train",
)

# Re-exports — public surface for the simple rules.
lora_base_model = _lora_base_model
lora_corpus = _lora_corpus
lora_dataset = _lora_dataset
lora_recipe = _lora_recipe

# Provenance: `lora_lineage(target = ...)` emits the transitive
# dataset/recipe/base lineage of a train/merge/adapter target as JSON;
# `lora_lineage_aspect` is exposed for consumers wiring their own audits.
lora_lineage = _lora_lineage
lora_lineage_aspect = _lora_lineage_aspect

# Default RunPod image + GPU. Used by `lora_train` when
# `backend = "runpod"` and no explicit override is passed.
_DEFAULT_RUNPOD_GPU = "NVIDIA H100 80GB HBM3"
_DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

def lora_train(
        name,
        base,
        recipe,
        dataset,
        backend = "runpod",
        # RunPod-only knobs:
        runpod_gpu = None,
        runpod_image = None,
        runpod_cloud = "SECURE",
        # Optional RunPod network volume. When `data_volume` is set, the
        # dataset is staged to the volume via S3 and read from the mount
        # instead of the slow SSH workdir rsync (which also drops
        # genrule-built datasets). `data_center` must match the volume's
        # data center (e.g. "EU-RO-1") — the pod is placed there.
        data_volume = "",
        data_center = "",
        # Empty = no wandb. Set to the W&B project name (e.g.
        # "agora", "rules_agentic_ide") to enable W&B tracking
        # for backend = "runpod" runs. The pod will pip-install
        # wandb, forward WANDB_API_KEY from the local env, log
        # in, and torchtune's metric_logger becomes WandBLogger.
        wandb_project = "",
        visibility = None):
    """Declare a LoRA training run.

    Always emits the typed `lora.v1.TrainingJobSpec` JSON
    (`<name>.jobspec.json`). When `backend == "runpod"`, also
    composes with `@rules_runpod` to produce:

      * `<name>_runpod_manifest_toml` — synthesized runpod TOML
        (build action via `write_file`).
      * `<name>_runpod_manifest` — typed manifest target.
      * `<name>_runpod_job` — typed job spec + `bazel run`-able
        `<name>_runpod_job.run` sibling.

    The synthesized manifest's `setup` installs torchtune and the
    `run` block is a v0 placeholder; the real torchtune wiring
    lands in rules_lora v0.0.3 (`runtime/torchtune_runner/` becomes
    a runnable entrypoint that reads the jobspec).

    Args:
      name: target name.
      base: label to a `lora_base_model` target.
      recipe: label to a `lora_recipe` target.
      dataset: label to a `lora_dataset` target.
      backend: one of "local" | "runpod" | "modal".
      runpod_gpu: override the default RunPod GPU type. Either a single
        type (string) or an ordered fallback list — the runpod-cli tries
        each in turn, advancing past capacity ("no instances available")
        errors. e.g. `["NVIDIA L40S", "NVIDIA A40", "NVIDIA A100 80GB PCIe"]`.
      runpod_image: override the default RunPod image.
      visibility: standard bazel visibility.
    """
    _lora_train_rule(
        name = name,
        base = base,
        recipe = recipe,
        dataset = dataset,
        backend = backend,
        visibility = visibility,
    )

    if backend == "local":
        _lora_local_runner_rule(
            name = name + "_local_runner_script",
            adapter_name = name,
            recipe = recipe,
            dataset = dataset,
            base = base,
            visibility = ["//visibility:private"],
        )
        sh_binary(
            name = name + ".run",
            srcs = [":" + name + "_local_runner_script"],
            data = [
                recipe,
                dataset,
                "@rules_lora//runtime/local_runner:local_runner.sh",
            ],
            deps = ["@bazel_tools//tools/bash/runfiles"],
            visibility = visibility,
        )
        return

    if backend != "runpod":
        return

    # `runpod_gpu` accepts a single GPU type (string) or an ordered
    # fallback list. The list reaches the manifest as `gpu_type = [...]`;
    # runpod-cli tries each in turn, advancing past capacity errors.
    if runpod_gpu == None:
        gpus = [_DEFAULT_RUNPOD_GPU]
    elif type(runpod_gpu) == "string":
        gpus = [runpod_gpu]
    else:
        gpus = runpod_gpu
    pod_type = gpus[0]  # primary — jobspec metadata; manifest carries the full list
    image = runpod_image or _DEFAULT_RUNPOD_IMAGE

    # v0.0.4: the manifest TOML is synthesized by the Rust binary in
    # //runtime/runpod_orchestrator. setup installs torchtune + the
    # HF CLI and pre-fetches the base model; run renders an effective
    # torchtune config from the rule's recipe attrs and invokes
    # `tune run lora_finetune_single_device`. Replaces the v0.0.2
    # `echo placeholder` and its `write_file` synth.
    # 0.0.24: the manifest synth reads the dataset's source_path
    # from LoraDatasetInfo to bake an explicit DATASET=<path> into
    # the run script (replaces the v0.0.23 find-based discovery
    # that silently failed when the workspace had multiple .jsonls).
    _lora_runpod_manifest_synth(
        name = name + "_runpod_manifest_toml",
        adapter_name = name,
        recipe = recipe,
        base = base,
        dataset = dataset,
        gpu_type = gpus,
        image = image,
        cloud_type = runpod_cloud,
        wandb_project = wandb_project,
        network_volume_id = data_volume,
        data_center = data_center,
        visibility = ["//visibility:private"],
    )

    runpod_manifest(
        name = name + "_runpod_manifest",
        src = ":" + name + "_runpod_manifest_toml",
        workdir = ".",
        # The synthesized run script writes to
        # `$(pwd)/outputs/adapter-<name>` so the path runpod-cli's
        # post-train rsync looks for matches it. v0.0.15 had this as
        # `["adapter-<name>"]` which silently failed the pull.
        outputs = ["outputs/adapter-" + name],
        visibility = visibility,
    )

    runpod_job(
        name = name + "_runpod_job",
        manifest = ":" + name + "_runpod_manifest",
        pod_type = pod_type,
        image = image,
        # Single-shot training: tear down the pod on success or
        # failure. Without this, every `bazel run` that errors mid-
        # tune leaves an orphan A100 burning $1.20/hr until manually
        # deleted. The adapter is pulled to outputs/ before the
        # failure-terminate fires, so partial checkpoints come back.
        ephemeral = True,
        visibility = visibility,
    )

def lora_merge(
        name,
        adapter_dir,
        base,
        out_dir,
        push_repo = "",
        private = True,
        visibility = None):
    """Fold a trained LoRA adapter into its base and export an HF dir.

    Emits:
      * `<name>` — the rule; carries `LoraBaseModelInfo(id = push_repo)`
        so it can be used directly as a `lora_train(base = ...)`.
      * `<name>.run` — `bazel run`-able; merges `outputs/adapter-…` into
        the base (candle, CPU), writes `out_dir`, and — when `push_repo`
        is set — pushes the merged dir to the HF hub via the `hf` CLI.

    The adapter is a runtime artifact (training pulls it to
    `outputs/adapter-<train-name>`), so `adapter_dir`/`out_dir` are
    workspace-relative path strings resolved at run time. To use the
    result as a training base, set `push_repo` and `bazel run :<name>.run`
    before the dependent `lora_train` run.

    Args:
      name: target name.
      adapter_dir: workspace-relative trained-adapter dir.
      base: label to the `lora_base_model` the adapter trained on.
      out_dir: workspace-relative output dir for the merged model.
      push_repo: optional HF repo id to push the merged model to.
      private: create the HF repo as private when pushing (default True).
      visibility: standard bazel visibility.
    """
    _lora_merge_rule(
        name = name,
        adapter_dir = adapter_dir,
        base = base,
        out_dir = out_dir,
        push_repo = push_repo,
        private = private,
        visibility = visibility,
    )
    sh_binary(
        name = name + ".run",
        srcs = [":" + name],
        data = ["@rules_lora//runtime/lora_merge:lora-merge"],
        deps = ["@bazel_tools//tools/bash/runfiles"],
        visibility = visibility,
    )

def expert_manifest(
        name,
        adapters,
        routing = "nearest_centroid",
        cluster_manifest = None,
        visibility = None):
    """Bundle N trained adapters into an `ExpertManifest.binpb`.

    Wire shape mirrors `[[rules_agentic_ide]]`'s
    `agentic_ide.v1.ExpertManifest`. v0.0.1: placeholder filegroup;
    v0.0.2 emits the real binpb.
    """
    native.filegroup(
        name = name,
        srcs = adapters,
        visibility = visibility,
    )
    _ = routing
    _ = cluster_manifest
