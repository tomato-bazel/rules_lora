# rules_lora

Bazel-native LoRA fine-tuning: typed SFT datasets, declarative recipes, runpod/local training backends, ExpertManifest emit

## Status: v0.0.1 — scaffold

No public surface yet. See `CHANGELOG.md` for what has shipped.

## Install

`.bazelrc`:

```
common --registry=https://raw.githubusercontent.com/fastverk/bazel-registry/main/
common --registry=https://bcr.bazel.build/
```

`MODULE.bazel`:

```python
bazel_dep(name = "rules_lora", version = "0.0.1")
```
