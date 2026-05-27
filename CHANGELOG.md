# Changelog

All notable changes to rules_lora. The format is loosely
[Keep a Changelog](https://keepachangelog.com/) — version headers
mirror the published bazel-registry entries (when we publish; for
now this repo is premium / private).

## 0.0.22 — Pin local-backend deps to a known-good triangle

The torchtune / torchao / kagglehub / kagglesdk dep graph breaks
under several unpinned permutations on Apple-Silicon-MPS:

  * Latest torchao (0.13+) requires `torch>=2.11`.
  * Latest torchtune imports `from kagglesdk.kaggle_env import
    get_web_endpoint`, which the 0.1.x kagglesdk on PyPI doesn't
    export.
  * Latest kagglehub depends on a kagglesdk that breaks the import.
  * torchtune 0.3.x doesn't yet have the import path issue but
    pre-dates `lora_qwen2_1_5b` we use.

Pin the local install to the May-2026 known-good set:
  * `torchao==0.7.0`
  * `torchtune==0.5.0`
  * `kagglehub<0.3`
  * `torch` — let pip pick the latest matching version.

The RunPod backend continues to pin `torchao==0.5.0` +
`torchtune==0.3.1` in its own setup (matched to torch 2.4 in the
runpod/pytorch image).

## 0.0.21 — Local backend prefers python 3.11

torchtune's transitive deps (kagglehub → kagglesdk) hit import-time
breakage under Python 3.14 (`cannot import name 'get_web_endpoint'`).
Pick `python3.11` if available (the ML stack's lingua franca),
falling back to `python3.12` then `python3`. On macOS with
`brew install python@3.11` this picks the brew interpreter
automatically.

## 0.0.20 — Local backend: install torch + unpin versions

Two local-backend fixes uncovered by the agora smoke run:

* Install `torch` explicitly in the venv (the previous pip install
  list assumed torch was already present, as in the RunPod
  pytorch image). Without it the MPS detection that runs `import
  torch` always falls through to CPU.

* Unpin `torchao` and `torchtune` versions for the local install.
  v0.0.17's pin to `torchao==0.5.0` is not published for
  aarch64-apple-darwin (`pip` returns 0.7.0 as the floor). Let pip
  pick the latest matching set per platform; the RunPod image
  still pins to torch 2.4-compatible versions in its own
  `setup`.

* Move device detection *after* the pip install so `import torch`
  succeeds and MPS gets picked on Apple Silicon.

## 0.0.19 — `exec bash $RUNNER` (no +x needed)

`exports_files` doesn't stamp the executable bit on shell scripts;
v0.0.17/18's generated wrapper did `exec "$RUNNER"` which failed
with `Permission denied`. Switch to `exec bash "$RUNNER"` —
bash reads the shebang directly without needing the exec bit.

## 0.0.18 — Fix `_runfiles_path` for external-repo short_paths

Cleanup-release of 0.0.17. `_runfiles_path(file, ctx)` returned
`<workspace>/<short_path>` unconditionally; for external-repo files
whose `short_path` already starts with `../<canonical>/...`, this
produced runfiles paths like `rules_lora+/../rules_lora+/runtime/...`
that `rlocation` couldn't resolve. Strip the leading `../` for
external-short-path inputs.

## 0.0.17 — `lora_train(backend = "local")` real entrypoint

The previously-stubbed `local` backend now emits a runnable
`<name>.run` sh_binary that drives torchtune on the host:

  * `runtime/local_runner/local_runner.sh` — venv bootstrap (workspace-
    local `.venvs/lora-local/`), `pip install torchao==0.5.0 torchtune
    ==0.3.1 + HF tooling`, HF base-model fetch, inline torchtune
    config render with auto-detected `device` (`mps` on macOS,
    `cuda` if `nvidia-smi`, else `cpu`), `tune run
    lora_finetune_single_device --config <yaml>`.
  * `_lora_local_runner` private rule reads LoraRecipeInfo +
    LoraBaseModelInfo + LoraDatasetInfo and generates an entry
    script that passes the hyperparams positionally to the runner.
  * `lora_train(backend = "local")` wires the rule into a `<name>.run`
    sh_binary.

Adapter lands at `outputs/adapter-<name>/` in the user's workspace
— no rsync, no pod, no orphan A100s. For 9-row LoRA fine-tunes
the on-laptop MPS path is the right default; reserve the runpod
backend for serious data volume.

## 0.0.16 — Manifest `outputs` path matches torchtune save dir

v0.0.15 declared the manifest's `outputs = ["adapter-<name>"]`
but the synthesized run script saves to
`$(pwd)/outputs/adapter-<name>`. The mismatch silently failed
runpod-cli's post-train rsync-back: the adapter trained, the pod
terminated (ephemeral), but the local `outputs/` ended up empty.
Fix: outputs becomes `["outputs/adapter-<name>"]`.

## 0.0.15 — Drop `o_proj` default + ephemeral pod

Two paper-iteration QoL fixes:

* `lora_recipe.target_modules` default loses `o_proj`. torchtune's
  `tune_to_peft_adapter_config` doesn't accept `o_proj` as a target
  module for Qwen2 / Llama3, so post-train peft conversion errored
  (with the adapter weights themselves saved fine). New default
  `["q_proj", "k_proj", "v_proj"]` round-trips through the peft
  config save.

* `lora_train(backend = "runpod")` now passes `ephemeral = True` to
  the emitted `runpod_job`. The `.run` wrapper threads
  `--down-on-success --down-on-failure` through runpod-cli, so an
  errored `tune run` no longer leaves an orphan A100 burning
  $1.20/hr until manually deleted. Bumps min rules_runpod dep to
  0.0.5 (where `--down-on-failure` lands).

## 0.0.14 — Drop unsupported model arg

`lora_qwen2_1_5b` doesn't accept `apply_lora_to_output`. Remove
it; add `lora_dropout: 0.0` for parity with the builder's default.

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
