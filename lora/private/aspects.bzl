"""rules_lora provenance aspect.

`lora_lineage_aspect` threads a `LoraLineageInfo` up the
`dataset -> recipe -> base -> train -> merge` graph, collecting each node's
provenance automatically — no hand-maintained sha plumbing. `lora_lineage`
materializes the collected lineage as a JSON manifest you can attach to a
release, diff across runs, or feed an audit.

Idiomatic shape: a single aspect over the typed `*Info` providers + the
attribute edges, plus a thin rule that runs it and emits the artifact.
"""

load(
    ":providers.bzl",
    "LoraAdapterInfo",
    "LoraBaseModelInfo",
    "LoraDatasetInfo",
    "LoraLineageInfo",
    "LoraRecipeInfo",
)

# The attribute edges training provenance flows along. The aspect both
# propagates over these (attr_aspects) and reads them to merge transitive
# lineage.
_EDGE_ATTRS = ["base", "recipe", "dataset", "adapter"]

def _own_record(target):
    """The provenance record for `target` itself, or None if it carries no
    rules_lora provider (e.g. a plain `lora_train` orchestration node — its
    value is the deps it threads)."""
    if LoraDatasetInfo in target:
        i = target[LoraDatasetInfo]
        return struct(
            kind = "dataset",
            label = str(target.label),
            detail = "schema={};source={};schema_n={}".format(i.schema, i.source_path, i.n_examples),
        )
    if LoraRecipeInfo in target:
        i = target[LoraRecipeInfo]
        return struct(
            kind = "recipe",
            label = str(target.label),
            detail = "framework={};rank={};alpha={};lr={};micro_bs={};grad_accum={};epochs={}".format(
                i.framework,
                i.rank,
                i.alpha,
                i.learning_rate,
                i.micro_batch_size,
                i.grad_accum_steps,
                i.epochs,
            ),
        )
    if LoraBaseModelInfo in target:
        i = target[LoraBaseModelInfo]
        return struct(
            kind = "base",
            label = str(target.label),
            detail = "id={};revision={}".format(i.id, i.revision),
        )
    if LoraAdapterInfo in target:
        i = target[LoraAdapterInfo]
        return struct(
            kind = "adapter",
            label = str(target.label),
            detail = "recipe_sha={};dataset_sha={};val_loss={}".format(
                i.recipe_sha,
                i.dataset_sha,
                i.val_loss,
            ),
        )
    return None

def _lora_lineage_aspect_impl(target, ctx):
    transitive = []
    for attr_name in _EDGE_ATTRS:
        dep = getattr(ctx.rule.attr, attr_name, None)

        # Single-label edges only; skip unset attrs and label_lists.
        if dep != None and type(dep) != "list" and LoraLineageInfo in dep:
            transitive.append(dep[LoraLineageInfo].records)
    direct = []
    rec = _own_record(target)
    if rec != None:
        direct.append(rec)
    return [LoraLineageInfo(records = depset(direct = direct, transitive = transitive))]

lora_lineage_aspect = aspect(
    implementation = _lora_lineage_aspect_impl,
    attr_aspects = _EDGE_ATTRS,
    provides = [LoraLineageInfo],
    doc = "Collects transitive training provenance into LoraLineageInfo.",
)

def _lora_lineage_impl(ctx):
    records = ctx.attr.target[LoraLineageInfo].records.to_list()

    # Sorted for deterministic output (records are a depset; order is not
    # guaranteed). json.encode handles escaping of labels / HF ids.
    rows = sorted(
        [{"kind": r.kind, "label": r.label, "detail": r.detail} for r in records],
        key = lambda r: (r["kind"], r["label"]),
    )
    out = ctx.actions.declare_file(ctx.label.name + ".lineage.json")
    ctx.actions.write(out, json.indent(json.encode(rows), indent = "  ") + "\n")
    return [DefaultInfo(files = depset([out]))]

lora_lineage = rule(
    implementation = _lora_lineage_impl,
    attrs = {
        "target": attr.label(
            mandatory = True,
            aspects = [lora_lineage_aspect],
            doc = "A lora_train / lora_merge / adapter target whose provenance to trace.",
        ),
    },
    doc = "Emit `<name>.lineage.json` — the transitive dataset/recipe/base provenance of `target`.",
)
