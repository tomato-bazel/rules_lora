# rules_lora

Bazel-native **LoRA fine-tuning**: datasets, training recipes, base models and
training runs declared as Bazel targets, with the trained adapter staged back
into the workspace.

The dataset and recipe are build artifacts, validated at build time (schema,
minimum example count, base-model compatibility). Training itself runs on a
backend — your machine, a [RunPod](https://www.runpod.io) GPU pod (via
[rules_runpod](https://github.com/tomato-bazel/rules_runpod)), or Modal
(not yet implemented) — when you `bazel run` the train target.

## Setup

`rules_lora` is published to the
[tomato-bazel registry](https://github.com/tomato-bazel/bazel-registry).
Add it ahead of the BCR in your `.bazelrc`:

```
common --registry=https://registry.tbzl.dev/
common --registry=https://bcr.bazel.build/

# rules_lora declares an isolated crate universe (`use_extension(...,
# isolate = True)`); Bazel rejects that without this flag.
common --experimental_isolated_extension_usages
```

Then in `MODULE.bazel`:

```starlark
bazel_dep(name = "rules_lora", version = "0.1.5")
```

## Usage

```starlark
load(
    "@rules_lora//lora:defs.bzl",
    "lora_base_model",
    "lora_dataset",
    "lora_recipe",
    "lora_train",
)

lora_dataset(
    name = "cluster_9_sft",
    src = "training/shards/cluster_009/sft.jsonl",
    schema = "messages_v1",
    min_examples = 50,
)

lora_recipe(
    name = "rank16_a32_3epoch",
    framework = "torchtune",
    rank = 16,
    alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    learning_rate = "2e-4",
    micro_batch_size = 4,
    grad_accum_steps = 8,
    epochs = 3,
)

lora_base_model(
    name = "qwen_0_5b",
    repo = "Qwen/Qwen2.5-0.5B-Instruct",
    revision = "main",
)

lora_train(
    name = "cluster_9",
    base = ":qwen_0_5b",
    recipe = ":rank16_a32_3epoch",
    dataset = ":cluster_9_sft",
)
```

`bazel build :cluster_9` writes the typed `lora.v1.TrainingJobSpec`
(`cluster_9.jobspec.json`); `bazel run :cluster_9.run` trains it and pulls the
adapter to `outputs/adapter-cluster_9`. [`examples/smoke`](examples/smoke) uses
every macro.

| macro | what it declares |
|---|---|
| `lora_dataset` | A typed SFT dataset (JSONL), validated at build time. |
| `lora_corpus` | A dataset *produced* by running your transform binary over a `source` filegroup, chained from upstream corpora via `deps`. |
| `lora_recipe` | A declarative training recipe: LoRA hyperparameters and training-loop knobs, rendered to a torchtune config at run time. |
| `lora_base_model` | A Hugging Face hub model pinned by repo and revision. |
| `lora_train` | The training run. `:<name>` is the jobspec; `:<name>.run` executes it on the selected backend. |
| `lora_merge` | Folds a trained adapter into its base (CPU, via candle) and exports a standalone HF model dir, optionally pushed to the hub. |
| `lora_lineage` | Emits the transitive dataset / recipe / base lineage of a train, merge or adapter target as JSON. |
| `expert_manifest` | Bundles N trained adapters into an `ExpertManifest` that maps each routing cluster to its adapter (placeholder filegroup today). |

### Backends

The backend is a **toolchain**, selected by platform rather than by a
per-target attribute. rules_lora registers all three; RunPod is the default.

```
bazel run :cluster_9.run                                                # runpod
bazel run :cluster_9.run --platforms=@rules_lora//lora/backend:local_platform
bazel run :cluster_9.run --platforms=@rules_lora//lora/backend:modal_platform
```

The RunPod backend reads the RunPod credentials described in the
[rules_runpod README](https://github.com/tomato-bazel/rules_runpod#credentials).
`lora_train` also takes `runpod_gpu` (a type, or an ordered fallback list),
`runpod_image`, `data_volume` / `data_center` (stage the dataset on a network
volume) and `wandb_project` (opt-in Weights & Biases tracking; forwards
`WANDB_API_KEY` from your environment). `HF_TOKEN` is forwarded to the pod too.

## Layout

```
lora/
  defs.bzl              public macros
  backend/              backend toolchain type, constraints, platforms, toolchains
  private/              rules, providers, toolchain rule, lineage aspect
runtime/
  runpod_orchestrator/  Rust: composes the jobspec and the RunPod manifest
  local_runner/         Python: local (CPU / Apple-Silicon MPS) training orchestrator
  torchtune_runner/     Python: JSONL validation and torchtune recipe rendering
  lora_merge/           Rust: folds an adapter into its base, exports an HF dir
proto/
  lora.proto            TrainingJobSpec and TrainingArtifact wire schema
examples/
  smoke/                every macro, analysis-only
  corpus_smoke/         a `lora_corpus` chain with a trivial transform
```

## Non-goals

- **Full pretraining or full fine-tuning.** LoRA only; full fine-tuning is a
  different recipe family with different infrastructure costs.
- **Hermetic training inside a Bazel action.** Training runs on a GPU (or MPS)
  host that Bazel can't sandbox. The recipe and dataset are Bazel artifacts and
  the trained adapter is staged back afterwards.
  [`docs/hermetic-runners-roadmap.md`](docs/hermetic-runners-roadmap.md) covers
  what is and isn't pinned today.
- **Bit-exact reproducibility.** An adapter is pinned by recipe, dataset and
  base-model revision, but exact-bit results depend on backend determinism
  (cuDNN and friends), which isn't promised.

## License

MIT. See [LICENSE](LICENSE). Changes are recorded in
[CHANGELOG.md](CHANGELOG.md).
