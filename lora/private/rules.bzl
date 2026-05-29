"""rules_lora private rule implementations.

The macros in //lora:defs.bzl are thin wrappers that pick a backend
and instantiate these rules. Keeping the rule defs private lets us
evolve the public surface without touching every consumer.
"""

load(":providers.bzl",
     "LoraAdapterInfo",
     "LoraBaseModelInfo",
     "LoraDatasetInfo",
     "LoraRecipeInfo")

# ============================================================
# lora_dataset — validate + sha-pin a JSONL.
# ============================================================
def _lora_dataset_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".jsonl")
    sha_out = ctx.actions.declare_file(ctx.label.name + ".sha")
    args = ctx.actions.args()
    args.add("--in", ctx.file.src.path)
    args.add("--out", out.path)
    args.add("--sha-out", sha_out.path)
    args.add("--schema", ctx.attr.schema)
    args.add("--min-examples", ctx.attr.min_examples)
    ctx.actions.run(
        executable = ctx.executable._validator,
        inputs = [ctx.file.src],
        outputs = [out, sha_out],
        arguments = [args],
        mnemonic = "LoraDatasetValidate",
        progress_message = "Validating LoRA dataset %s" % ctx.label,
    )
    return [
        DefaultInfo(files = depset([out, sha_out])),
        LoraDatasetInfo(
            jsonl = out,
            source_path = ctx.file.src.short_path,
            schema = ctx.attr.schema,
            # n_examples and sha are emitted into sha_out as a JSON
            # sidecar; build-time consumers read it via a separate
            # action when they need the actual values.
            n_examples = -1,  # placeholder — real value in the sidecar
            sha = "",         # placeholder — real value in the sidecar
        ),
    ]

lora_dataset = rule(
    implementation = _lora_dataset_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".jsonl"]),
        "schema": attr.string(
            default = "messages_v1",
            values = ["messages_v1", "instruction_v1"],
        ),
        "min_examples": attr.int(default = 1),
        "_validator": attr.label(
            default = "@rules_lora//runtime/torchtune_runner:validate_jsonl",
            executable = True,
            cfg = "exec",
        ),
    },
)

# ============================================================
# lora_recipe — render a YAML config (torchtune/axolotl/peft).
# ============================================================
def _lora_recipe_impl(ctx):
    yaml = ctx.actions.declare_file(ctx.label.name + ".yaml")
    args = ctx.actions.args()
    args.add("--framework", ctx.attr.framework)
    args.add("--rank", ctx.attr.rank)
    args.add("--alpha", ctx.attr.alpha)
    args.add_joined("--target-modules", ctx.attr.target_modules, join_with = ",")
    args.add("--learning-rate", ctx.attr.learning_rate)
    args.add("--micro-batch-size", ctx.attr.micro_batch_size)
    args.add("--grad-accum-steps", ctx.attr.grad_accum_steps)
    args.add("--epochs", ctx.attr.epochs)
    args.add("--out", yaml.path)
    ctx.actions.run(
        executable = ctx.executable._renderer,
        outputs = [yaml],
        arguments = [args],
        mnemonic = "LoraRecipeRender",
        progress_message = "Rendering LoRA recipe %s" % ctx.label,
    )
    return [
        DefaultInfo(files = depset([yaml])),
        LoraRecipeInfo(
            yaml = yaml,
            framework = ctx.attr.framework,
            rank = ctx.attr.rank,
            alpha = ctx.attr.alpha,
            target_modules = ctx.attr.target_modules,
            learning_rate = ctx.attr.learning_rate,
            micro_batch_size = ctx.attr.micro_batch_size,
            grad_accum_steps = ctx.attr.grad_accum_steps,
            epochs = ctx.attr.epochs,
            sha = "",  # filled by the renderer's sidecar
        ),
    ]

lora_recipe = rule(
    implementation = _lora_recipe_impl,
    attrs = {
        "framework": attr.string(
            default = "torchtune",
            values = ["torchtune", "axolotl", "peft", "trl"],
        ),
        "rank": attr.int(default = 16),
        "alpha": attr.int(default = 32),
        "target_modules": attr.string_list(
            # Drop `o_proj`: torchtune's tune_to_peft_adapter_config
            # (called at checkpoint save) doesn't recognize `o_proj`
            # as a target module for Qwen2/Llama3, so any post-train
            # peft conversion errors. q/k/v give the same effective
            # rank for the attention block in practice.
            default = ["q_proj", "k_proj", "v_proj"],
        ),
        "learning_rate": attr.string(default = "2e-4"),
        "micro_batch_size": attr.int(default = 4),
        "grad_accum_steps": attr.int(default = 8),
        "epochs": attr.int(default = 3),
        "_renderer": attr.label(
            default = "@rules_lora//runtime/torchtune_runner:render_recipe",
            executable = True,
            cfg = "exec",
        ),
    },
)

# ============================================================
# lora_base_model — pin an HF hub model by revision.
# ============================================================
def _lora_base_model_impl(ctx):
    return [
        DefaultInfo(),
        LoraBaseModelInfo(
            id = ctx.attr.repo,
            revision = ctx.attr.revision,
            config_path = ctx.file.config if ctx.file.config else None,
        ),
    ]

lora_base_model = rule(
    implementation = _lora_base_model_impl,
    attrs = {
        "repo": attr.string(mandatory = True, doc = "HF hub repo id."),
        "revision": attr.string(
            mandatory = True,
            doc = "Commit sha or `sha256:<digest>` of the model snapshot.",
        ),
        "config": attr.label(
            allow_single_file = [".json"],
            doc = "Optional local model_config.json override.",
        ),
    },
)

# ============================================================
# lora_train — orchestrate a training run.
# ============================================================
#
# `bazel run :<name>.train` (the wrapper macro emits a `_train`
# executable target). The rule itself produces the adapter file as
# a build output if the backend supports declarative outputs (local
# CPU smoke). For remote backends (runpod), the build produces a
# job-spec file and the executable does the run + download.
def _lora_train_impl(ctx):
    spec = ctx.actions.declare_file(ctx.label.name + ".jobspec.json")
    args = ctx.actions.args()
    args.add("write-jobspec")
    args.add("--name", ctx.label.name)
    args.add("--recipe", ctx.attr.recipe[LoraRecipeInfo].yaml.path)
    args.add("--dataset", ctx.attr.dataset[LoraDatasetInfo].jsonl.path)
    args.add("--base-id", ctx.attr.base[LoraBaseModelInfo].id)
    args.add("--base-revision", ctx.attr.base[LoraBaseModelInfo].revision)
    args.add("--backend", ctx.attr.backend)
    args.add("--out", spec.path)
    ctx.actions.run(
        executable = ctx.executable._spec_writer,
        inputs = [
            ctx.attr.recipe[LoraRecipeInfo].yaml,
            ctx.attr.dataset[LoraDatasetInfo].jsonl,
        ],
        outputs = [spec],
        arguments = [args],
        mnemonic = "LoraJobSpec",
        progress_message = "Composing LoRA job spec for %s" % ctx.label,
    )
    return [
        DefaultInfo(files = depset([spec])),
    ]

lora_train = rule(
    implementation = _lora_train_impl,
    attrs = {
        "base": attr.label(
            mandatory = True,
            providers = [LoraBaseModelInfo],
        ),
        "recipe": attr.label(
            mandatory = True,
            providers = [LoraRecipeInfo],
        ),
        "dataset": attr.label(
            mandatory = True,
            providers = [LoraDatasetInfo],
        ),
        "backend": attr.string(
            default = "runpod",
            values = ["local", "runpod", "modal"],
        ),
        "_spec_writer": attr.label(
            default = "@rules_lora//runtime/runpod_orchestrator:write_jobspec",
            executable = True,
            cfg = "exec",
        ),
    },
)

# ============================================================
# _lora_local_runner — emit a wrapper shell script for the
#   `lora_train(backend = "local")` path. v0.0.17.
# ============================================================
#
# The wrapper reads the recipe + base providers, then writes a
# small shell script whose entry point execs the rules_lora-side
# `runtime/local_runner/local_runner.sh` with all of the hyperparams
# hardcoded as positional args. local_runner.sh handles venv setup,
# HF model download, torchtune config rendering, and the `tune run`
# itself.
#
# Why a generated script (not a sh_binary args list directly):
# providers aren't readable in macros, so we need a rule to extract
# `recipe.rank`, `recipe.alpha`, etc. The rule's output is fed to a
# sh_binary in the macro layer as its `srcs`.

def _runfiles_path(file, ctx):
    """Compute the file's runfiles-relative path.

    Files from the main repository have `short_path` like
    `training/parser_recipe.yaml` and a runfiles path of
    `<main_ws>/training/parser_recipe.yaml`.

    Files from external repos have `short_path` like
    `../rules_lora+/runtime/local_runner/local_runner.sh` (the
    `../` escapes the main-repo subtree); the canonical runfiles
    path strips the leading `../`.
    """
    short = file.short_path
    if short.startswith("../"):
        return short[len("../"):]
    return ctx.workspace_name + "/" + short

def _lora_local_runner_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".sh")
    recipe_info = ctx.attr.recipe[LoraRecipeInfo]
    base_info = ctx.attr.base[LoraBaseModelInfo]
    dataset_info = ctx.attr.dataset[LoraDatasetInfo]

    target_modules_csv = ",".join(recipe_info.target_modules)
    recipe_rl = _runfiles_path(recipe_info.yaml, ctx)
    dataset_rl = _runfiles_path(dataset_info.jsonl, ctx)
    runner_rl = _runfiles_path(ctx.file._runner_script, ctx)

    script_content = """#!/usr/bin/env bash
# Generated by //lora/private:rules.bzl::_lora_local_runner_impl.
# Entry point for `bazel run :<name>.run` when
# `lora_train(backend = "local")`. Hardcodes the recipe hyperparams
# read from LoraRecipeInfo + LoraBaseModelInfo, then exec's
# rules_lora's runtime/local_runner/local_runner.sh.

set -uo pipefail

# --- begin runfiles.bash initialization v3 ---
set +e
f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${{RUNFILES_DIR:-/dev/null}}/$f" 2>/dev/null || \\
  source "$(grep -sm1 "^$f " "${{RUNFILES_MANIFEST_FILE:-/dev/null}}" | cut -f2- -d' ')" 2>/dev/null || \\
  source "$0.runfiles/$f" 2>/dev/null || \\
  source "$(dirname "$0").runfiles/$f" 2>/dev/null || \\
  source "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")").runfiles/$f" 2>/dev/null || \\
  {{ echo>&2 "ERROR: cannot find $f (runfiles library missing)"; exit 1; }}
f=
set -e
# --- end runfiles.bash initialization v3 ---

RUNNER="$(rlocation "{runner_rl}")"
# `exec bash $RUNNER` (not `exec $RUNNER`) because Bazel doesn't
# stamp the +x bit on `exports_files`-d shell scripts; bash reads
# the shebang and runs without needing the exec permission.
exec bash "$RUNNER" \\
    "{recipe_rl}" \\
    "{dataset_rl}" \\
    "{adapter_name}" \\
    "{base_id}" \\
    "{base_revision}" \\
    "{family}" \\
    "{rank}" \\
    "{alpha}" \\
    "{target_modules}" \\
    "{learning_rate}" \\
    "{micro_batch_size}" \\
    "{grad_accum_steps}" \\
    "{epochs}" \\
    "$@"
""".format(
        runner_rl = runner_rl,
        recipe_rl = recipe_rl,
        dataset_rl = dataset_rl,
        adapter_name = ctx.attr.adapter_name,
        base_id = base_info.id,
        base_revision = base_info.revision,
        family = ctx.attr.family,
        rank = recipe_info.rank,
        alpha = recipe_info.alpha,
        target_modules = target_modules_csv,
        learning_rate = recipe_info.learning_rate,
        micro_batch_size = recipe_info.micro_batch_size,
        grad_accum_steps = recipe_info.grad_accum_steps,
        epochs = recipe_info.epochs,
    )

    ctx.actions.write(output = out, content = script_content, is_executable = True)
    return [DefaultInfo(files = depset([out]))]

lora_local_runner = rule(
    implementation = _lora_local_runner_impl,
    attrs = {
        "adapter_name": attr.string(mandatory = True),
        "recipe": attr.label(mandatory = True, providers = [LoraRecipeInfo]),
        "dataset": attr.label(mandatory = True, providers = [LoraDatasetInfo]),
        "base": attr.label(mandatory = True, providers = [LoraBaseModelInfo]),
        "family": attr.string(
            default = "qwen2",
            values = ["qwen2", "llama3", "mistral"],
        ),
        "_runner_script": attr.label(
            default = "@rules_lora//runtime/local_runner:local_runner.sh",
            allow_single_file = True,
        ),
    },
)

# ============================================================
# _lora_merge — fold a trained adapter into its base + export. v0.0.34.
# ============================================================
#
# Emits a `bazel run`-able wrapper that invokes the
# //runtime/lora_merge:lora-merge binary (candle merge, CPU) on a
# trained adapter dir, writing a standalone HF model dir and optionally
# pushing it to the hub via the `hf` CLI.
#
# The adapter is a *runtime* artifact (training pulls it to
# `outputs/adapter-<name>`), not a Bazel output — so `adapter_dir` and
# `out_dir` are workspace-relative path strings, resolved against
# `$BUILD_WORKSPACE_DIRECTORY` at run time (same place training writes
# `outputs/`). The rule also emits `LoraBaseModelInfo(id = push_repo)`
# so the merged model can be used directly as a `lora_train(base = ...)`
# — provided you `bazel run :<name>.run` (merge + push) first.
def _lora_merge_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".sh")
    base_info = ctx.attr.base[LoraBaseModelInfo]
    tool_rl = _runfiles_path(ctx.executable._merge_tool, ctx)

    push_line = ""
    if ctx.attr.push_repo:
        push_line = '    --push-repo "{repo}" {private} \\\n'.format(
            repo = ctx.attr.push_repo,
            private = "--private" if ctx.attr.private else "",
        )

    script_content = """#!/usr/bin/env bash
# Generated by //lora/private:rules.bzl::_lora_merge_impl.
# Entry point for `bazel run :<name>.run` (lora_merge). Folds the
# trained adapter into its base via runtime/lora_merge:lora-merge.

set -uo pipefail

# --- begin runfiles.bash initialization v3 ---
set +e
f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${{RUNFILES_DIR:-/dev/null}}/$f" 2>/dev/null || \\
  source "$(grep -sm1 "^$f " "${{RUNFILES_MANIFEST_FILE:-/dev/null}}" | cut -f2- -d' ')" 2>/dev/null || \\
  source "$0.runfiles/$f" 2>/dev/null || \\
  source "$(dirname "$0").runfiles/$f" 2>/dev/null || \\
  source "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")").runfiles/$f" 2>/dev/null || \\
  {{ echo>&2 "ERROR: cannot find $f (runfiles library missing)"; exit 1; }}
f=
set -e
# --- end runfiles.bash initialization v3 ---

MERGE="$(rlocation "{tool_rl}")"
# Resolve adapter_dir / out_dir against the source tree (where training
# pulled the adapter to outputs/), not the runfiles sandbox.
cd "${{BUILD_WORKSPACE_DIRECTORY:-$PWD}}"
exec "$MERGE" \\
    --adapter "{adapter_dir}" \\
    --base-id "{base_id}" \\
    --base-revision "{base_revision}" \\
    --out "{out_dir}" \\
{push_line}    "$@"
""".format(
        tool_rl = tool_rl,
        adapter_dir = ctx.attr.adapter_dir,
        base_id = base_info.id,
        base_revision = base_info.revision,
        out_dir = ctx.attr.out_dir,
        push_line = push_line,
    )

    ctx.actions.write(output = out, content = script_content, is_executable = True)
    return [
        DefaultInfo(files = depset([out])),
        # The merged model lives at push_repo on the hub once `.run` has
        # been executed; expose it as a base so downstream lora_train can
        # reference this target directly.
        LoraBaseModelInfo(
            id = ctx.attr.push_repo,
            revision = "main",
            config_path = None,
        ),
    ]

lora_merge = rule(
    implementation = _lora_merge_impl,
    attrs = {
        "adapter_dir": attr.string(
            mandatory = True,
            doc = "Workspace-relative dir of the trained adapter " +
                  "(e.g. `outputs/adapter-<train-name>`).",
        ),
        "out_dir": attr.string(
            mandatory = True,
            doc = "Workspace-relative output dir for the merged model.",
        ),
        "base": attr.label(
            mandatory = True,
            providers = [LoraBaseModelInfo],
            doc = "The `lora_base_model` the adapter was trained on.",
        ),
        "push_repo": attr.string(
            doc = "If set, push the merged dir to this HF repo id " +
                  "(via the `hf` CLI). Also becomes this target's " +
                  "LoraBaseModelInfo.id so it can be a lora_train base.",
        ),
        "private": attr.bool(
            default = True,
            doc = "Create the HF repo as private when pushing.",
        ),
        "_merge_tool": attr.label(
            default = "@rules_lora//runtime/lora_merge:lora-merge",
            executable = True,
            cfg = "exec",
        ),
    },
)

# ============================================================
# _lora_runpod_manifest_synth — synthesize the pod-side TOML.
# ============================================================
#
# Replaces the v0.0.2 hand-rolled `write_file` placeholder with a
# real torchtune-invoking manifest, rendered by the Rust binary in
# //runtime/runpod_orchestrator. The rule reads LoraRecipeInfo +
# LoraBaseModelInfo providers and forwards the relevant fields to
# the orchestrator's `write-runpod-manifest` subcommand.
#
# `family` selects the torchtune model component (qwen2 / llama3 /
# mistral). The default `qwen2` matches the LoRA-trained NL→capability
# parser in agora — extend the match arm in the orchestrator when a
# new family lands.
def _lora_runpod_manifest_synth_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".toml")
    recipe = ctx.attr.recipe[LoraRecipeInfo]
    base = ctx.attr.base[LoraBaseModelInfo]
    # Workspace-relative path to the underlying source JSONL of the
    # `lora_dataset`. Used by the synth template to bake an explicit
    # `DATASET=<path>` into the run script (replaces the v0.0.23
    # `find`-based discovery that silently failed).
    dataset_src_path = ctx.attr.dataset[LoraDatasetInfo].source_path
    dataset_jsonl = ctx.attr.dataset[LoraDatasetInfo].jsonl

    args = ctx.actions.args()
    args.add("write-runpod-manifest")
    args.add("--name", ctx.attr.adapter_name)
    args.add("--dataset-src", dataset_src_path)
    # Network-volume data path (optional). When a volume is set the
    # manifest mounts it and reads the dataset from it; we stage the
    # validated JSONL from its run-time bazel-bin location via S3.
    if ctx.attr.network_volume_id:
        args.add("--network-volume-id", ctx.attr.network_volume_id)
        args.add("--data-center", ctx.attr.data_center)
        args.add("--volume-mount", ctx.attr.volume_mount)
        args.add("--stage-local", "bazel-bin/" + dataset_jsonl.short_path)
    # `before_each` repeats the flag per value (`--gpu-type A --gpu-type B`),
    # matching the orchestrator's clap `Vec<String>`. Plain
    # `add_all("--gpu-type", …)` would emit the flag once + bare values.
    args.add_all(ctx.attr.gpu_type, before_each = "--gpu-type")
    args.add("--image", ctx.attr.image)
    args.add("--base-id", base.id)
    args.add("--base-revision", base.revision)
    args.add("--family", ctx.attr.family)
    args.add("--cloud-type", ctx.attr.cloud_type)
    args.add("--rank", recipe.rank)
    args.add("--alpha", recipe.alpha)
    args.add_joined("--target-modules", recipe.target_modules, join_with = ",")
    args.add("--learning-rate", recipe.learning_rate)
    args.add("--micro-batch-size", recipe.micro_batch_size)
    args.add("--grad-accum-steps", recipe.grad_accum_steps)
    args.add("--epochs", recipe.epochs)
    args.add("--wandb-project", ctx.attr.wandb_project)
    args.add("--out", out.path)

    # Depend on the validated JSONL so `bazel run`-ing the job builds it
    # to bazel-bin, where the manifest's `stage` step uploads it from.
    ctx.actions.run(
        executable = ctx.executable._synth,
        inputs = [dataset_jsonl],
        outputs = [out],
        arguments = [args],
        mnemonic = "LoraRunpodManifestSynth",
        progress_message = "Synthesizing runpod manifest for %s" % ctx.label,
    )
    return [DefaultInfo(files = depset([out]))]

lora_runpod_manifest_synth = rule(
    implementation = _lora_runpod_manifest_synth_impl,
    attrs = {
        "adapter_name": attr.string(
            mandatory = True,
            doc = "Adapter name used in `lora-<name>` job + output paths.",
        ),
        "recipe": attr.label(mandatory = True, providers = [LoraRecipeInfo]),
        "base": attr.label(mandatory = True, providers = [LoraBaseModelInfo]),
        # Reads `source_path` from LoraDatasetInfo to bake the
        # dataset path into the run script. The validated bazel-bin
        # artifact (`jsonl`) is skipped by rsync (`bazel-*` exclude),
        # so the source-tree path is what survives the upload.
        "dataset": attr.label(mandatory = True, providers = [LoraDatasetInfo]),
        "gpu_type": attr.string_list(
            mandatory = True,
            doc = "Ordered RunPod GPU-type fallback list. The runpod-cli " +
                  "tries each in turn, advancing on a capacity error.",
        ),
        "image": attr.string(mandatory = True),
        "family": attr.string(
            default = "qwen2",
            values = ["qwen2", "llama3", "mistral"],
            doc = "torchtune model family — selects tokenizer + model component.",
        ),
        "cloud_type": attr.string(
            default = "SECURE",
            values = ["COMMUNITY", "SECURE"],
            doc = (
                "RunPod cloud tier. COMMUNITY is cheaper but often " +
                "exhausted; SECURE is the right default for paper-" +
                "iteration runs where availability matters."
            ),
        ),
        # Empty = no wandb (StdoutLogger only, current default).
        # Set to the wandb project name to enable: the manifest
        # forwards WANDB_API_KEY from the local env, the pod-side
        # setup pip-installs wandb + `wandb login`s, and the
        # rendered torchtune metric_logger becomes WandBLogger.
        "wandb_project": attr.string(default = ""),
        # Optional network-volume data path. When `network_volume_id`
        # is set, the manifest mounts the volume, stages the dataset to
        # it via S3, reads the dataset from the mount, and drops the
        # heavy `training/full` corpus from the workdir rsync.
        "network_volume_id": attr.string(default = ""),
        "data_center": attr.string(default = ""),
        "volume_mount": attr.string(default = "/workspace"),
        "_synth": attr.label(
            default = "@rules_lora//runtime/runpod_orchestrator:runpod_orchestrator",
            executable = True,
            cfg = "exec",
        ),
    },
)

# ============================================================
# lora_corpus — typed corpus rule.
# ============================================================
#
# A `lora_corpus` is a `lora_dataset` produced by running a
# user-supplied `transform` binary over a `source` filegroup,
# optionally chained from upstream corpora declared via `deps`.
#
# The transform contract (CLI):
#   * Repeated  --input <path>       once per file in `source`.
#   * Repeated  --corpus-dep <path>  once per validated upstream
#                                    `lora_corpus` / `lora_dataset` JSONL.
#   * Single    --output <path>      where to write the SFT JSONL.
#
# Consumers see a `LoraDatasetInfo`, so `lora_corpus` plugs in
# anywhere `lora_dataset` is accepted (notably as the `dataset`
# attr of `lora_train`).
#
# Versioning: v0.0.3 introduces this rule. Corpus deps form a DAG
# (Bazel enforces no cycles); transitive deps are propagated only
# through Bazel's build graph — the rule itself flattens to the
# direct deps when invoking the transform (the upstream corpora's
# transforms have already run and produced their JSONL artifacts).
def _lora_corpus_impl(ctx):
    raw_out = ctx.actions.declare_file(ctx.label.name + ".raw.jsonl")
    val_out = ctx.actions.declare_file(ctx.label.name + ".jsonl")
    sha_out = ctx.actions.declare_file(ctx.label.name + ".sha")

    # ─── Step 1: invoke the user-supplied transform. ───
    transform_args = ctx.actions.args()
    transform_inputs = []
    for src in ctx.files.source:
        transform_args.add("--input", src.path)
        transform_inputs.append(src)
    for dep in ctx.attr.deps:
        dep_jsonl = dep[LoraDatasetInfo].jsonl
        transform_args.add("--corpus-dep", dep_jsonl.path)
        transform_inputs.append(dep_jsonl)
    transform_args.add("--output", raw_out.path)
    ctx.actions.run(
        executable = ctx.executable.transform,
        inputs = transform_inputs,
        outputs = [raw_out],
        arguments = [transform_args],
        mnemonic = "LoraCorpusTransform",
        progress_message = "Transforming corpus %s" % ctx.label,
    )

    # ─── Step 2: validate the transform output. ───
    # Reuses the same validator as `lora_dataset` so the schema /
    # min_examples / sha contract is identical across both rules.
    val_args = ctx.actions.args()
    val_args.add("--in", raw_out.path)
    val_args.add("--out", val_out.path)
    val_args.add("--sha-out", sha_out.path)
    val_args.add("--schema", ctx.attr.schema)
    val_args.add("--min-examples", ctx.attr.min_examples)
    ctx.actions.run(
        executable = ctx.executable._validator,
        inputs = [raw_out],
        outputs = [val_out, sha_out],
        arguments = [val_args],
        mnemonic = "LoraCorpusValidate",
        progress_message = "Validating corpus %s" % ctx.label,
    )

    return [
        DefaultInfo(files = depset([val_out, sha_out])),
        LoraDatasetInfo(
            jsonl = val_out,
            # lora_corpus is a derived target — no single source
            # JSONL to upload. The runpod backend errors clearly if
            # someone wires a lora_corpus directly into lora_train
            # with backend="runpod". For now, route the corpus
            # through an explicit `lora_dataset(src=<corpus_output>)`
            # if you need runpod ingest.
            source_path = "",
            schema = ctx.attr.schema,
            n_examples = -1,
            sha = "",
        ),
    ]

lora_corpus = rule(
    implementation = _lora_corpus_impl,
    attrs = {
        "source": attr.label_list(
            allow_files = True,
            doc = "Raw source files fed to the transform as repeated --input flags.",
        ),
        "transform": attr.label(
            mandatory = True,
            executable = True,
            cfg = "exec",
            doc = (
                "Binary that reads repeated `--input` / `--corpus-dep` " +
                "flags and writes one SFT JSONL to `--output`."
            ),
        ),
        "deps": attr.label_list(
            providers = [LoraDatasetInfo],
            doc = (
                "Upstream `lora_corpus` or `lora_dataset` targets fed " +
                "to the transform via repeated `--corpus-dep` flags."
            ),
        ),
        "schema": attr.string(
            default = "messages_v1",
            values = ["messages_v1", "instruction_v1"],
        ),
        "min_examples": attr.int(default = 1),
        "_validator": attr.label(
            default = "@rules_lora//runtime/torchtune_runner:validate_jsonl",
            executable = True,
            cfg = "exec",
        ),
    },
)
