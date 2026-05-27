"""rules_lora public API.

```starlark
load("@rules_lora//lora:defs.bzl",
     "lora_base_model",
     "lora_dataset",
     "lora_recipe",
     "lora_train",
     "expert_manifest")
```

* `lora_dataset`, `lora_recipe`, `lora_base_model` — thin
  re-exports of the underlying rules.
* `lora_train` — v0.0.2: now a macro. Always emits the typed
  jobspec; additionally composes with `@rules_runpod` when
  `backend = "runpod"` to emit a synthesized manifest +
  `runpod_job`, giving the user `bazel run :<name>.runpod_job.run`.
* `expert_manifest` — bundle N adapters as the routing input.

Macros forward to rules in `//lora/private:rules.bzl`, which fill
providers from `//lora/private:providers.bzl`.
"""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_runpod//runpod:defs.bzl", "runpod_job", "runpod_manifest")
load(
    "//lora/private:rules.bzl",
    _lora_base_model = "lora_base_model",
    _lora_corpus = "lora_corpus",
    _lora_dataset = "lora_dataset",
    _lora_recipe = "lora_recipe",
    _lora_runpod_manifest_synth = "lora_runpod_manifest_synth",
    _lora_train_rule = "lora_train",
)

# Re-exports — public surface for the simple rules.
lora_base_model = _lora_base_model
lora_corpus = _lora_corpus
lora_dataset = _lora_dataset
lora_recipe = _lora_recipe

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
      runpod_gpu: override the default RunPod GPU type.
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

    if backend != "runpod":
        return

    pod_type = runpod_gpu or _DEFAULT_RUNPOD_GPU
    image = runpod_image or _DEFAULT_RUNPOD_IMAGE

    # v0.0.4: the manifest TOML is synthesized by the Rust binary in
    # //runtime/runpod_orchestrator. setup installs torchtune + the
    # HF CLI and pre-fetches the base model; run renders an effective
    # torchtune config from the rule's recipe attrs and invokes
    # `tune run lora_finetune_single_device`. Replaces the v0.0.2
    # `echo placeholder` and its `write_file` synth.
    _lora_runpod_manifest_synth(
        name = name + "_runpod_manifest_toml",
        adapter_name = name,
        recipe = recipe,
        base = base,
        gpu_type = pod_type,
        image = image,
        cloud_type = runpod_cloud,
        visibility = ["//visibility:private"],
    )

    runpod_manifest(
        name = name + "_runpod_manifest",
        src = ":" + name + "_runpod_manifest_toml",
        workdir = ".",
        outputs = ["adapter-" + name],
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

def expert_manifest(name, adapters, routing = "nearest_centroid",
                    cluster_manifest = None, visibility = None):
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
