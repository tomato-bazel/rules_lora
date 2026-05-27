"""rules_lora Starlark providers.

Four typed surfaces shared across all backends. The providers are
the contract; each rule fills them out, each backend consumes them.
"""

LoraDatasetInfo = provider(
    doc = "Validated SFT dataset.",
    fields = {
        "jsonl": "File — the validated JSONL artifact.",
        "source_path": (
            "str — workspace-relative short_path of the underlying " +
            "source JSONL (`ctx.file.src.short_path`). Used by " +
            "runpod backend to bake an explicit DATASET= into the " +
            "run script. Source-tree path is needed because the " +
            "validated bazel-bin jsonl is excluded from rsync upload."
        ),
        "schema": "str — one of {`messages_v1`, `instruction_v1`}.",
        "n_examples": "int — recorded at validate time.",
        "sha": "str — BLAKE3 hex of the JSONL bytes; pins the dataset.",
    },
)

LoraRecipeInfo = provider(
    doc = "A frozen training recipe.",
    fields = {
        "yaml": "File — rendered torchtune / axolotl / peft YAML.",
        "framework": "str — `torchtune` | `axolotl` | `peft` | `trl`.",
        "rank": "int — LoRA rank.",
        "alpha": "int — LoRA alpha.",
        "target_modules": "list[str] — attention modules adapted.",
        "learning_rate": "str — optimizer LR (kept as string so e.g. `2e-4` survives).",
        "micro_batch_size": "int.",
        "grad_accum_steps": "int.",
        "epochs": "int.",
        "sha": "str — BLAKE3 hex of the rendered YAML.",
    },
)

LoraBaseModelInfo = provider(
    doc = "A pinned base model.",
    fields = {
        "id": "str — HF hub id (e.g. `google/gemma-3-2b-it`).",
        "revision": "str — HF hub commit sha or sha256 digest.",
        "config_path": "File or None — optional local model_config.json override.",
    },
)

LoraAdapterInfo = provider(
    doc = "A trained LoRA adapter artifact.",
    fields = {
        "safetensors": "File — the adapter weights.",
        "adapter_config": "File — adapter_config.json (peft-compatible).",
        "base_model": "LoraBaseModelInfo — the base this adapter targets.",
        "recipe_sha": "str — provenance: which recipe trained this.",
        "dataset_sha": "str — provenance: which data trained this.",
        "val_loss": "float — final eval loss (or NaN if not measured).",
    },
)

ExpertManifestInfo = provider(
    doc = "The routing contract: cluster_id -> adapter mapping.",
    fields = {
        "binpb": "File — agentic_ide.v1.ExpertManifest serialized.",
        "n_experts": "int.",
    },
)
