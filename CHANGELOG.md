# Changelog

All notable changes to rules_lora. The format is loosely
[Keep a Changelog](https://keepachangelog.com/) — version headers
mirror the published bazel-registry entries (when we publish; for
now this repo is premium / private).

## 0.0.13 — Use size-specific torchtune model builder

`torchtune.models.qwen2.lora_qwen2` requires every architectural arg
(vocab_size / num_heads / embed_dim / …) explicitly. Switch the
qwen2 family to `torchtune.models.qwen2.lora_qwen2_1_5b`, the
torchtune-shipped builder that bakes the Qwen2/2.5-1.5B
architecture in. Hardcoded to 1.5B for v0; v0.0.14 generalizes via
a `--family-variant` flag.

## 0.0.12 — Round out the torchtune YAML config

torchtune 0.3.1's `lora_finetune_single_device` recipe rejected
the v0.0.11 config with `Missing key max_steps_per_epoch`. Add
all of the keys torchtune unconditionally reads at recipe init:
  * `max_steps_per_epoch: null`
  * `resume_from_checkpoint: False`
  * `save_adapter_weights_only: True`
  * `enable_activation_checkpointing: False`
  * `lr_scheduler` block (cosine warmup, 1 step warmup)
  * `optimizer.fused: True`
  * `profiler` block (disabled)

## 0.0.11 — Pin torchao + torchtune versions

Latest torchao (0.13+) imports torch's `int1` dtype, which doesn't
exist in torch 2.4 (the version in runpod/pytorch:2.4.0). Pin to
`torchao==0.5.0` + `torchtune==0.3.1` — last release where both
play nice with torch 2.4. Bumping the runpod image is a v0.0.12
follow-up.

## 0.0.10 — Add `torchao` to pod-side setup

torchtune now imports `torchao` unconditionally on package import.
Without it `tune run` exits with `ModuleNotFoundError: No module
named 'torchao'`. Adds `torchao` to the setup's pip install.

## 0.0.9 — Fix format string in v0.0.8 (yanked)

v0.0.8 shipped a Rust `format!` template with an unescaped `{`
inside a comment of the bash heredoc; the orchestrator binary
failed to compile. 0.0.9 swaps the offending JSON-snippet in the
comment for a prose description.

## 0.0.8 — Pod-side dataset auto-detect by content sniff (yanked)

v0.0.4–v0.0.7's pod-side `run` block tried to find the SFT JSONL via
naming heuristic (`*lora_dataset*` or `dataset.jsonl`). That missed
common conventions like `training/sft.jsonl` and meant consumers
either renamed their seed file or wired in a bazel-bin path the
workdir rsync doesn't see.

The detector now walks every `.jsonl` in the workdir (skipping
`bazel-*` dirs) and picks the first whose head bytes contain
`"messages"` — i.e. a real `messages_v1` row. Robust to whatever
the consumer named the file. Failure message also names the
matched candidate count so debugging is one ssh away.

## 0.0.7 — Default RunPod image tag fix

v0.0.4 pinned `runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04`
as the default RunPod image — but that tag was never published to
Docker Hub, so RunPod's container daemon failed pod startup with
`manifest unknown`. Bump default to `2.4.0-py3.11-cuda12.4.1-devel-
ubuntu22.04` — same Python/CUDA/Ubuntu stack, but a real tag (used
by prime-transformer's working manifests).

## 0.0.6 — `lora_train(runpod_cloud = "SECURE")` knob

Adds a `runpod_cloud` attr to the `lora_train` macro (default
`"SECURE"`). Threads through to the synthesized manifest's
`[resources].cloud_type`. SECURE is the right default for
paper-iteration runs — COMMUNITY tier is frequently exhausted
for popular GPU types (H100 / A100 / A40) and the resulting
`create_pod: HTTP 500: There are no instances currently available`
error is a poor first-run experience. Override to `"COMMUNITY"`
when cost matters more than availability.

## 0.0.5 — runpod manifest TOML structure fix

v0.0.4 emitted a TOML where `setup` and `run` followed `[resources]`,
so the TOML parser folded them inside that table and runpod-cli
rejected the manifest with `missing field setup`. Fix: top-level
keys (name, workdir, outputs, setup, run) now come *before* the
`[resources]` table.

## 0.0.4 — `lora_train` pod-side manifest invokes real torchtune

The v0.0.2 `write_file` placeholder (`run = """echo placeholder"""`)
is replaced with `lora_runpod_manifest_synth`, a private rule that
calls the Rust binary `//runtime/runpod_orchestrator
write-runpod-manifest`. The Rust binary reads `LoraRecipeInfo` +
`LoraBaseModelInfo` and renders a manifest TOML whose `setup` and
`run` blocks:

  * Install `torchtune`, `huggingface_hub[cli]`, `transformers`,
    `datasets` on the pod's pre-baked pytorch image.
  * Pre-fetch the base model (revision-pinned) and stash the
    cached path for the train step.
  * Render an effective torchtune config YAML inline by
    interpolating the LoRA hyperparams + per-job paths.
  * Invoke `tune run lora_finetune_single_device --config ...` —
    the real torchtune LoRA fine-tune loop.
  * Drop the adapter at `outputs/adapter-<name>/`.

Supported model families today (selected by a `family` attr on the
synth rule, default `qwen2`):

  * Qwen2.5 family (`qwen2`)
  * Llama 3 family (`llama3`)
  * Mistral family (`mistral`)

Each maps to the matching torchtune tokenizer + lora model
component. Extending the matrix is a single match arm in the Rust
binary plus the corresponding tokenizer convention.

`LoraRecipeInfo` gains three new fields propagated from the
`lora_recipe` rule attrs:

  * `learning_rate: str`     (kept as string so `2e-4` survives)
  * `micro_batch_size: int`
  * `grad_accum_steps: int`

These were previously rendered only into the recipe YAML; the
manifest synth needs them as structured attrs.

Smoke at `examples/smoke/`:

    bazel build //examples/smoke:smoke_jobspec_runpod_manifest_toml
    # renders a TOML whose `run` block has real `tune run` invocation
    # instead of the v0.0.2 placeholder.

## 0.0.3 — `lora_corpus` rule with corpus-DAG deps

New public macro `lora_corpus`: declare an SFT dataset *produced by
running a user-supplied transform binary over a `source` filegroup,
chained from upstream corpora via `deps`*. Three consumers
(rules_agentic_ide chat traces, agora capability-auction corpus,
the NDA'd third) share the input/transform/validate/output skeleton;
the rule factors it out.

Public surface added:

* `lora_corpus(name, source, transform, deps, schema, min_examples)`
  — runs the transform once with repeated `--input` / `--corpus-dep`
  / single `--output` flags, then runs the existing
  `validate_jsonl` validator on the transform output. Returns
  `LoraDatasetInfo`, so it plugs in anywhere `lora_dataset` is
  accepted (in particular as the `dataset` attr of `lora_train`).

Corpus deps form a DAG. The rule itself flattens to *direct* deps
when invoking the transform (the upstream corpora's transforms
have already produced their validated JSONL artifacts as build
outputs); Bazel's build graph enforces no cycles and propagates
transitive rebuilds.

Smoke at `examples/corpus_smoke/`:

    bazel build //examples/corpus_smoke:derived_corpus
    # base_corpus: 3 examples
    # derived_corpus: 6 examples (3 source + 3 from dep)

Also includes the v0.0.2 features (deferred from registry release):
`lora_train` macro composes with `@rules_runpod` when
`backend = "runpod"`, auto-emitting `<name>_runpod_job.run`.

## 0.0.1 — scaffold + public API frozen

Public surface (`@rules_lora//lora:defs.bzl`):

* `lora_dataset` — typed SFT-JSONL dataset, validated + sha-pinned
  at build time. Schemas: `messages_v1` (OpenAI chat format),
  `instruction_v1`.
* `lora_recipe` — declarative training recipe. Frameworks:
  `torchtune` (rendered), `axolotl` (rendered), `peft` + `trl`
  (TODO templates).
* `lora_base_model` — HF hub model pinned by repo + revision.
* `lora_train` — composes the inputs into a
  `lora.v1.TrainingJobSpec` (JSON, build-time) that a backend
  executes at `bazel run` time.
* `expert_manifest` — bundles N adapters into the
  `agentic_ide.v1.ExpertManifest.binpb` shape (placeholder
  filegroup in v0.0.1; real rule in v0.0.2).

Runtime stubs:

* `runtime/torchtune_runner/{validate_jsonl, render_recipe}.py` —
  build-time tools, std-lib only.
* `runtime/runpod_orchestrator/` (Rust) — `write-jobspec`
  subcommand functional; `run` subcommand pending v0.1.

Smoke at `examples/smoke/` exercises all four macros end-to-end
and produces a self-contained jobspec — `bazel build
//examples/smoke:smoke_jobspec` is the regression test.
