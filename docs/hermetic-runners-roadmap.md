# Hermetic LoRA runners — roadmap

Status: **deferred to a dedicated, on-hardware session.** This is the one piece
of the rules_lora runner work that cannot be verified without a GPU/MPS box and
a live RunPod account, so it is written up here rather than half-built. A hollow
scaffold (stubbed pod lifecycle, fake torch lock) is *worse* than the working
venv + `@rules_runpod` paths that ship today — don't merge one.

## Where we are now (shipped)

The backend dispatch + de-shell work already landed (PR #1 on `main`):

- **Per-platform backend toolchain** (`//lora/backend`): `local` / `runpod` /
  `modal` are registered toolchains selected by `--platforms`; `lora_train`
  resolves the toolchain (`LoraBackendInfo`) for the jobspec composer + backend
  identity. The `.run` entry dispatches on the same `:backend` constraint via
  `select()`.
- **No generated shell of ours.** The local + merge run entries are `py_binary`
  orchestrators reading build-generated JSON configs (`*.local.json`,
  `*.merge.json`); `local_runner.sh` and the bash wrappers are gone.
- **What is still non-hermetic** (by design, deferred here):
  - `runtime/local_runner/local_train.py` is a thin orchestrator — it creates a
    runtime venv, `pip install`s torch/torchtune, downloads the HF model, and
    shells `tune run`. Network + host accelerator; not hermetic.
  - `runtime/runpod_orchestrator/src/main.rs` — the `run` subcommand is a stub
    (`bail!("not implemented yet (v0.1)")`); the working RunPod path is the
    `@rules_runpod` macro composition.

## Goal

Two independent tracks, each ending in a runner **binary** the backend toolchain
points at, with no runtime venv and no `@rules_runpod` dependency.

---

## Track 1 — hermetic local runner (vendored torch)

Replace the runtime venv with Bazel-vendored deps + in-process torchtune.

1. **Lock the training deps.** Pin the working triangle (the comment set:
   `torch`, `torchtune==0.4.0`, `torchao==0.5.0`, `kagglehub<0.3`,
   `huggingface_hub[cli]`, `transformers`, `datasets`) into a
   `runtime/local_runner/requirements.txt` and compile a fully-resolved
   `requirements.lock`. **Landmine:** this triangle breaks on nearly every
   unpinned release; lock it once, on the target Python (3.11), and treat
   bumping it as a deliberate, tested change.
2. **`pip.parse` in `MODULE.bazel`** over the lock, exposing `@lora_pip//torch`,
   `@lora_pip//torchtune`, etc. Use **platform-conditional** requirement sets:
   MPS (mac arm64) vs CPU vs CUDA wheels are different downloads — `select()` the
   right `@lora_pip_{mps,cpu,cu121}//...` per `--platforms`. This is the part
   that needs care; analysis can check the labels resolve, only a real fetch +
   import confirms the wheel set.
3. **HF base model as a Bazel artifact.** A repository rule (or
   `http_file`/`http_archive`) that fetches the pinned `base_id@revision`
   snapshot into a repo, so the model is an input, not a runtime `hf download`.
   Large; cache-aware. (Or keep the runtime download as the explicit
   "non-hermetic edge" and document it — vendoring multi-GB models in Bazel is a
   real cost/benefit call.)
4. **In-process torchtune.** Rewrite `local_train.py` to `import torchtune` and
   invoke `lora_finetune_single_device` via its Python API against the rendered
   config — no `tune run` CLI subprocess, no venv activation. The `py_binary`
   then `deps` on the vendored torch/torchtune instead of building a venv.
5. **Wire it into the toolchain.** The `local` backend toolchain's runner becomes
   this hermetic `py_binary`; drop the venv path from `local_train.py`.

**Verification (needs hardware):** `bazel run` the local backend on an
Apple-Silicon box, confirm a real LoRA step trains end-to-end (device `mps`),
adapter lands in `outputs/adapter-<name>`. Repeat on a CUDA Linux box for the
`cu121` wheel set.

---

## Track 2 — orchestrator `run` subcommand (RunPod pod lifecycle)

Reimplement the pod lifecycle in `runtime/runpod_orchestrator` so the `runpod`
backend is a single binary, not an `@rules_runpod` macro composition.

1. **`run --jobspec <path>`** reads the typed `lora.v1.TrainingJobSpec`
   (composer already emits it; the jobspec carries `recipe_yaml`, base id/rev,
   backend, `backend_config_json`).
2. **Pod lifecycle against the RunPod API** (the work `@rules_runpod` does today):
   create pod (GPU-type fallback list, image, cloud tier, optional network
   volume + data-center placement) → stage dataset (S3/volume or SSH rsync) →
   `setup` (pip torchtune + HF CLI, prefetch base) → `run` (`tune run
   lora_finetune_single_device` with the synthesized config) → poll → pull
   `outputs/adapter-<name>` → **`ephemeral`: terminate on success *and* failure**
   (don't leak a paid GPU). Optional W&B forwarding.
3. **Drop `@rules_runpod`** from `lora_train`'s runpod branch; the `runpod`
   backend toolchain's runner becomes this orchestrator binary + the jobspec in
   runfiles. The `.run` `select()` then points at it.

**Verification (needs a live RunPod account):** `bazel run` the runpod backend,
watch a pod come up, train, return the adapter, and tear down — including the
failure path (kill mid-train, confirm no orphan GPU).

---

## Sequencing & guardrails

- The two tracks are independent; do Track 1 first (more self-contained, MPS is
  cheaper to iterate than RunPod GPUs).
- Keep the current working paths (venv local / `@rules_runpod`) in place until
  each replacement is verified on hardware — flip the toolchain runner only when
  green.
- Per the repo's DTO convention, the runner↔orchestrator contract is already a
  proto (`lora.v1.TrainingJobSpec`); keep new config on it.

## Related deferred items (not this doc's scope, noted for completeness)

- **rules_postgres Gate 3** runs only where the private `//crates/pipeline`
  clang/LLVM tools exist (the public `@rules_lang//rules/c` ships rule *defs*
  only). Making Gate 3 public would mean porting that ~8k-LOC Rust subsystem +
  clang toolchain into public rules_lang.
- **Stranded `atlas-v0.3.0` tag** on the Syntax-less polyglot commit (protected,
  can't delete; no release attached, unused). The live atlas is `atlas-v0.3.1`
  via `rules_lang 0.3.0`.
