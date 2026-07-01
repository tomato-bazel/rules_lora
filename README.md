# rules_lora

Bazel-native **LoRA fine-tuning** — hermetic, reproducible low-rank-adapter
training as Bazel targets.

## Use it

```starlark
# MODULE.bazel — resolves from the fastverk registry (registry.fastverk.com)
bazel_dep(name = "rules_lora", version = "0.0.35")
```

See the package `BUILD.bazel` / `defs.bzl` for the training + adapter rules.
Part of the tomato-bazel distribution.
