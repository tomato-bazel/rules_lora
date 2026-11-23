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
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_runpod//runpod:defs.bzl", "runpod_job", "runpod_manifest")
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
    _lora_local_config_rule = "lora_local_config",
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

# Default RunPod image + GPU for the runpod `.run` entry. These mirror the
# runpod backend toolchain's `default_gpu`/`default_image` (//lora/backend);
# they live here too because the runpod run entry is assembled at loading phase
# (`@rules_runpod`'s `runpod_job` is a macro) where the resolved toolchain
# isn't visible. The toolchain copy covers the build (jobspec) side.
_DEFAULT_RUNPOD_GPU = "NVIDIA H100 80GB HBM3"
_DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

def lora_train(
        name,
        base,
        recipe,
        dataset,
        # RunPod knobs (configure the runpod `.run` entry):
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
        # "my-lora-runs") to enable W&B tracking for the
        # runpod run. The pod will pip-install wandb, forward
        # WANDB_API_KEY from the local env, log in, and torchtune's
        # metric_logger becomes WandBLogger.
        wandb_project = "",
        visibility = None):
    """Declare a LoRA training run.

    The **backend is selected per-platform**, not per-target:

        bazel build :<name>                                  # default: runpod
        bazel build :<name> --platforms=@rules_lora//lora/backend:local_platform

    `:<name>` (the rule) resolves `@rules_lora//lora/backend:toolchain_type` and
    emits the typed `lora.v1.TrainingJobSpec` JSON (`<name>.jobspec.json`) tagged
    with the resolved backend. The runnable `:<name>.run` dispatches on the same
    `:backend` constraint via `select()`:

      * local  -> `:<name>_local` (host venv + torchtune; runtime/local_runner)
      * runpod -> `:<name>_runpod_job.run` (synthesized manifest + @rules_runpod)
      * modal  -> `:<name>_modal` (stub; not yet implemented)

    All three run entries are emitted regardless of platform (analysis-only for
    the inactive ones); the `select()` picks the one matching the build, so the
    run side and the toolchain-resolved build side agree on one platform choice.

    Args:
      name: target name.
      base: label to a `lora_base_model` target.
      recipe: label to a `lora_recipe` target.
      dataset: label to a `lora_dataset` target.
      runpod_gpu: override the default RunPod GPU type. Either a single
        type (string) or an ordered fallback list — the runpod-cli tries
        each in turn, advancing past capacity ("no instances available")
        errors. e.g. `["NVIDIA L40S", "NVIDIA A40", "NVIDIA A100 80GB PCIe"]`.
      runpod_image: override the default RunPod image.
      runpod_cloud: RunPod cloud type, `"SECURE"` (default) or
        `"COMMUNITY"`.
      data_volume: optional RunPod network volume ID. When set, the dataset
        is staged to the volume via S3 and read from the mount instead of
        rsynced over SSH.
      data_center: the network volume's data center (e.g. `"EU-RO-1"`);
        the pod is placed there. Required with `data_volume`.
      wandb_project: W&B project name. Empty (default) disables W&B; when
        set, the pod installs wandb, forwards `WANDB_API_KEY` from the local
        environment, and torchtune logs through `WandBLogger`.
      visibility: standard bazel visibility.
    """
    _lora_train_rule(
        name = name,
        base = base,
        recipe = recipe,
        dataset = dataset,
        visibility = visibility,
    )

    # ---- local backend run entry -------------------------------------------
    # The runner is a py_binary (runtime/local_runner:local_train) reading a
    # build-generated JSON config — no generated shell, no local_runner.sh.
    _lora_local_config_rule(
        name = name + "_local_config",
        adapter_name = name,
        recipe = recipe,
        dataset = dataset,
        base = base,
        visibility = ["//visibility:private"],
    )
    py_binary(
        name = name + "_local",
        srcs = ["@rules_lora//runtime/local_runner:local_train.py"],
        main = "local_train.py",
        args = ["--config", "$(rlocationpath :" + name + "_local_config)"],
        data = [
            ":" + name + "_local_config",
            dataset,
        ],
        deps = ["@rules_python//python/runfiles"],
        visibility = visibility,
    )

    # ---- runpod backend run entry ------------------------------------------
    # `runpod_gpu` accepts a single GPU type (string) or an ordered fallback
    # list. The list reaches the manifest as `gpu_type = [...]`; runpod-cli
    # tries each in turn, advancing past capacity errors.
    if runpod_gpu == None:
        gpus = [_DEFAULT_RUNPOD_GPU]
    elif type(runpod_gpu) == "string":
        gpus = [runpod_gpu]
    else:
        gpus = runpod_gpu
    pod_type = gpus[0]  # primary — jobspec metadata; manifest carries the full list
    image = runpod_image or _DEFAULT_RUNPOD_IMAGE

    # The manifest TOML is synthesized by the Rust binary in
    # //runtime/runpod_orchestrator. setup installs torchtune + the HF CLI and
    # pre-fetches the base model; run renders an effective torchtune config from
    # the recipe attrs and invokes `tune run lora_finetune_single_device`. The
    # synth reads the dataset's source_path from LoraDatasetInfo to bake an
    # explicit DATASET=<path> into the run script.
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
        # The synthesized run script writes to `$(pwd)/outputs/adapter-<name>`
        # so the path runpod-cli's post-train rsync looks for matches it.
        outputs = ["outputs/adapter-" + name],
        visibility = visibility,
    )
    runpod_job(
        name = name + "_runpod_job",
        manifest = ":" + name + "_runpod_manifest",
        pod_type = pod_type,
        image = image,
        # Single-shot training: tear down the pod on success or failure.
        # Without this, every `bazel run` that errors mid-tune leaves an orphan
        # A100 burning $1.20/hr until manually deleted. The adapter is pulled to
        # outputs/ before the failure-terminate fires, so partial checkpoints
        # come back.
        ephemeral = True,
        visibility = visibility,
    )

    # ---- modal backend run entry (stub) ------------------------------------
    write_file(
        name = name + "_modal_stub",
        out = name + "_modal_stub.py",
        content = [
            "import sys",
            "sys.exit(",
            "    'lora_train: the modal backend is not yet implemented.\\n'",
            "    'Select a working backend with one of:\\n'",
            "    '  --platforms=@rules_lora//lora/backend:local_platform\\n'",
            "    '  --platforms=@rules_lora//lora/backend:runpod_platform'",
            ")",
        ],
    )
    py_binary(
        name = name + "_modal",
        srcs = [":" + name + "_modal_stub"],
        main = name + "_modal_stub.py",
        visibility = visibility,
    )

    # ---- per-platform dispatch ---------------------------------------------
    # `:<name>.run` follows the build's `:backend` constraint to the matching
    # run entry — the same constraint the rule's toolchain resolution keys on.
    native.alias(
        name = name + ".run",
        actual = select({
            "@rules_lora//lora/backend:local": ":" + name + "_local",
            "@rules_lora//lora/backend:runpod": ":" + name + "_runpod_job.run",
            "@rules_lora//lora/backend:modal": ":" + name + "_modal",
            "//conditions:default": ":" + name + "_runpod_job.run",
        }),
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

    # The `.run` entry is a py_binary (runtime/lora_merge:merge_run) reading the
    # build-generated <name>.merge.json + subprocessing the lora-merge rust
    # binary — no generated shell.
    py_binary(
        name = name + ".run",
        srcs = ["@rules_lora//runtime/lora_merge:merge_run.py"],
        main = "merge_run.py",
        args = [
            "--config",
            "$(rlocationpath :" + name + ")",
            "--merge-tool",
            "$(rlocationpath @rules_lora//runtime/lora_merge:lora-merge)",
        ],
        data = [
            ":" + name,
            "@rules_lora//runtime/lora_merge:lora-merge",
        ],
        deps = ["@rules_python//python/runfiles"],
        visibility = visibility,
    )

def expert_manifest(
        name,
        adapters,
        routing = "nearest_centroid",  # @unused (placeholder until the real rule lands)
        cluster_manifest = None,  # @unused
        visibility = None):
    """Bundle N trained adapters into an `ExpertManifest.binpb`.

    The manifest maps each routing cluster to its adapter, built from
    the `TrainingArtifact`s in `//proto:lora.proto`.
    v0.0.1: placeholder filegroup; v0.0.2 emits the real binpb.
    """
    native.filegroup(
        name = name,
        srcs = adapters,
        visibility = visibility,
    )
