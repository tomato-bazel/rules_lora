//! runpod_orchestrator — build-time tooling for the runpod backend
//! of `lora_train`.
//!
//! Two subcommands:
//!   * `write-jobspec`         — serialize the rule's attrs into a
//!                               TrainingJobSpec JSON consumed by
//!                               downstream targets.
//!   * `write-runpod-manifest` — synthesize the runpod-cli manifest
//!                               TOML whose `setup` + `run` blocks
//!                               actually invoke torchtune to train
//!                               the LoRA adapter on the pod.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "runpod_orchestrator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Build-time: serialize the rule attrs into a TrainingJobSpec
    /// JSON file. The execute-time `run` subcommand reads this.
    WriteJobspec {
        #[arg(long)]
        name: String,
        #[arg(long)]
        recipe: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        base_id: String,
        #[arg(long)]
        base_revision: String,
        #[arg(long)]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build-time: synthesize the runpod-cli manifest TOML whose
    /// `setup` and `run` blocks pull HF weights, render an effective
    /// torchtune config from the recipe attrs, and invoke
    /// `tune run lora_finetune_single_device` on the pod.
    WriteRunpodManifest {
        /// Job name (e.g., `parser_jobspec`).
        #[arg(long)]
        name: String,
        /// Workspace-relative path to the dataset JSONL inside the
        /// rsync'd workdir. Set by the Starlark rule from
        /// `ctx.file.src.short_path` of the underlying `lora_dataset`.
        /// 0.0.24: replaces the legacy `find`-based dataset
        /// discovery in the run script, which would silently pick the
        /// wrong file (or no file) when multiple `.jsonl`s were in
        /// the workspace.
        #[arg(long)]
        dataset_src: String,
        /// RunPod GPU type(s), e.g. `NVIDIA H100 80GB HBM3`. Repeatable
        /// (`--gpu-type A --gpu-type B`): an ordered fallback list the
        /// runpod-cli tries in turn, advancing to the next on a
        /// capacity ("no instances available") error.
        #[arg(long)]
        gpu_type: Vec<String>,
        /// RunPod container image.
        #[arg(long)]
        image: String,
        /// HF hub repo id (e.g., `Qwen/Qwen2.5-1.5B-Instruct`).
        #[arg(long)]
        base_id: String,
        /// HF hub revision (commit sha or branch).
        #[arg(long)]
        base_revision: String,
        /// torchtune family (e.g., `qwen2`, `llama3`); selects the
        /// matching tokenizer + model_type component in the rendered
        /// config.
        #[arg(long, default_value = "qwen2")]
        family: String,
        /// RunPod cloud tier ("COMMUNITY" or "SECURE"). COMMUNITY is
        /// cheaper but often exhausted; SECURE is the right default
        /// for paper-iteration runs where availability matters.
        #[arg(long, default_value = "SECURE")]
        cloud_type: String,
        /// LoRA rank.
        #[arg(long)]
        rank: u32,
        /// LoRA alpha.
        #[arg(long)]
        alpha: u32,
        /// Comma-separated target modules.
        #[arg(long)]
        target_modules: String,
        /// Learning rate (rendered into the recipe).
        #[arg(long)]
        learning_rate: String,
        /// Micro batch size.
        #[arg(long)]
        micro_batch_size: u32,
        /// Gradient accumulation steps.
        #[arg(long)]
        grad_accum_steps: u32,
        /// Number of epochs.
        #[arg(long)]
        epochs: u32,
        /// Optional wandb project. When non-empty, the rendered
        /// manifest forwards WANDB_API_KEY, pip-installs wandb,
        /// and swaps torchtune's StdoutLogger for WandBLogger.
        #[arg(long, default_value = "")]
        wandb_project: String,
        /// Optional RunPod network volume id. When set, the manifest
        /// mounts the volume and reads the dataset from it instead of
        /// the rsync'd workdir — the fast data path that also delivers
        /// genrule-built datasets (which `bazel-*` rsync excludes drop).
        #[arg(long, default_value = "")]
        network_volume_id: String,
        /// RunPod data center of the volume (e.g. EU-RO-1). Required
        /// when `network_volume_id` is set — the pod must be there.
        #[arg(long, default_value = "")]
        data_center: String,
        /// Path where the network volume mounts on the pod.
        #[arg(long, default_value = "/workspace")]
        volume_mount: String,
        /// Run-time-resolvable local path to the dataset JSONL to stage
        /// onto the volume before deploy (e.g. `bazel-bin/<short_path>`).
        /// Empty = assume the dataset is already on the volume.
        #[arg(long, default_value = "")]
        stage_local: String,
        /// Destination path for the manifest TOML.
        #[arg(long)]
        out: PathBuf,
    },
    /// Execute-time: upload spec + dataset, poll the job, download
    /// the adapter. Not yet implemented; v0.1.
    Run {
        #[arg(long)]
        jobspec: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::WriteJobspec {
            name,
            recipe,
            dataset,
            base_id,
            base_revision,
            backend,
            out,
        } => write_jobspec(name, recipe, dataset, base_id, base_revision, backend, out),
        Cmd::WriteRunpodManifest {
            name,
            dataset_src,
            gpu_type,
            image,
            base_id,
            base_revision,
            family,
            cloud_type,
            rank,
            alpha,
            target_modules,
            learning_rate,
            micro_batch_size,
            grad_accum_steps,
            epochs,
            wandb_project,
            network_volume_id,
            data_center,
            volume_mount,
            stage_local,
            out,
        } => write_runpod_manifest(WriteRunpodManifestArgs {
            name,
            dataset_src,
            wandb_project,
            network_volume_id,
            data_center,
            volume_mount,
            stage_local,
            gpu_type,
            image,
            base_id,
            base_revision,
            family,
            cloud_type,
            rank,
            alpha,
            target_modules,
            learning_rate,
            micro_batch_size,
            grad_accum_steps,
            epochs,
            out,
        }),
        Cmd::Run { jobspec } => {
            anyhow::bail!(
                "runpod_orchestrator run: not implemented yet (v0.1). \
                 v0.0 only emits the spec at {}.",
                jobspec.display()
            )
        }
    }
}

fn write_jobspec(
    name: String,
    recipe: PathBuf,
    dataset: PathBuf,
    base_id: String,
    base_revision: String,
    backend: String,
    out: PathBuf,
) -> Result<()> {
    let recipe_bytes =
        std::fs::read(&recipe).with_context(|| format!("reading {}", recipe.display()))?;
    let dataset_bytes =
        std::fs::read(&dataset).with_context(|| format!("reading {}", dataset.display()))?;
    let recipe_sha = blake3::hash(&recipe_bytes).to_hex().to_string();
    let dataset_sha = blake3::hash(&dataset_bytes).to_hex().to_string();
    let recipe_yaml = String::from_utf8(recipe_bytes).context("recipe is not utf-8")?;

    let spec = serde_json::json!({
        "name": name,
        "recipe_sha": recipe_sha,
        "dataset_sha": dataset_sha,
        "base_model_id": base_id,
        "base_model_revision": base_revision,
        "backend": backend,
        "recipe_yaml": recipe_yaml,
        "backend_config_json": "{}",
        "max_minutes": 0,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&spec)?)
        .with_context(|| format!("writing {}", out.display()))?;
    eprintln!(
        "runpod_orchestrator: jobspec → {} (recipe={}…, dataset={}…)",
        out.display(),
        &recipe_sha[..12],
        &dataset_sha[..12]
    );
    Ok(())
}

// ─── write-runpod-manifest ──────────────────────────────────────────
//
// Synthesizes the runpod-cli manifest TOML. The `setup` and `run`
// blocks are real torchtune-invoking bash; placeholders (job name,
// HF model id, LoRA hyperparams) are interpolated at build time, the
// rest is static.
//
// Family → torchtune component mapping. v0 covers Qwen2 / Llama3 /
// Mistral; extension is a `match` arm + a tokenizer.json convention
// the family already uses upstream.

struct WriteRunpodManifestArgs {
    name: String,
    dataset_src: String,
    gpu_type: Vec<String>,
    image: String,
    base_id: String,
    base_revision: String,
    family: String,
    cloud_type: String,
    rank: u32,
    alpha: u32,
    target_modules: String,
    learning_rate: String,
    micro_batch_size: u32,
    grad_accum_steps: u32,
    epochs: u32,
    /// Optional wandb project. When set, the manifest forwards
    /// WANDB_API_KEY from the local env, `pip install wandb` runs
    /// in setup, and the rendered torchtune config swaps
    /// `StdoutLogger` for `WandBLogger` with this project name.
    /// Empty string = no wandb (StdoutLogger only).
    wandb_project: String,
    /// Network volume id to mount + read the dataset from. Empty =
    /// legacy workdir-rsync path.
    network_volume_id: String,
    data_center: String,
    volume_mount: String,
    stage_local: String,
    out: PathBuf,
}

fn write_runpod_manifest(a: WriteRunpodManifestArgs) -> Result<()> {
    if a.gpu_type.is_empty() {
        anyhow::bail!("at least one --gpu-type is required");
    }
    let target_modules_yaml = a
        .target_modules
        .split(',')
        .map(|s| format!("\"{}\"", s.trim()))
        .collect::<Vec<_>>()
        .join(", ");
    let family_components = match a.family.as_str() {
        "qwen2" => FamilyComponents {
            tokenizer: "torchtune.models.qwen2.qwen2_tokenizer",
            // Use the size-specific builder so we don't have to thread
            // every architectural arg (vocab_size / num_heads / …)
            // through the manifest. 1.5B is hardcoded for the agora
            // parser; v0.0.13 generalizes to a `--family-variant` arg.
            model_lora: "torchtune.models.qwen2.lora_qwen2_1_5b",
            checkpoint_model_type: "QWEN2",
        },
        "llama3" => FamilyComponents {
            tokenizer: "torchtune.models.llama3.llama3_tokenizer",
            model_lora: "torchtune.models.llama3.lora_llama3",
            checkpoint_model_type: "LLAMA3",
        },
        "mistral" => FamilyComponents {
            tokenizer: "torchtune.models.mistral.mistral_tokenizer",
            model_lora: "torchtune.models.mistral.lora_mistral",
            checkpoint_model_type: "MISTRAL",
        },
        other => anyhow::bail!(
            "unknown model family '{}'. Supported: qwen2 / llama3 / mistral.",
            other
        ),
    };

    let toml = render_manifest_toml(&a, &target_modules_yaml, &family_components);
    std::fs::write(&a.out, toml)
        .with_context(|| format!("writing {}", a.out.display()))?;
    eprintln!(
        "runpod_orchestrator: manifest → {} ({} adapter, family={})",
        a.out.display(),
        a.name,
        a.family
    );
    Ok(())
}

struct FamilyComponents {
    tokenizer: &'static str,
    model_lora: &'static str,
    checkpoint_model_type: &'static str,
}

fn render_manifest_toml(
    a: &WriteRunpodManifestArgs,
    target_modules_yaml: &str,
    fc: &FamilyComponents,
) -> String {
    // Wandb integration is opt-in by setting `wandb_project` on the
    // `lora_train` macro. When set:
    //   * `forward_envs = ["WANDB_API_KEY"]` propagates the local
    //     secret to the pod (runpod-cli's manifest schema field).
    //   * `pip install wandb` runs in setup.
    //   * `wandb login --relogin "$WANDB_API_KEY"` authenticates.
    //   * The rendered torchtune metric_logger swaps StdoutLogger
    //     for WandBLogger with the configured project + run name.
    // Render the GPU candidates as a TOML array; runpod-cli's manifest
    // schema accepts either a string or a list and tries them in order.
    let gpu_array = format!(
        "[{}]",
        a.gpu_type
            .iter()
            .map(|g| format!("\"{}\"", g.trim()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let wandb_enabled = !a.wandb_project.is_empty();
    // Always forward HF_TOKEN: the pod's `hf download` of the base model
    // needs it for private or gated repos (e.g. a merged two-stage base,
    // or Llama). runpod-cli skips any forward_envs var that isn't set
    // locally, so this is harmless when no token is present. WANDB_API_KEY
    // is added only when wandb tracking is enabled.
    let forward_list = if wandb_enabled {
        "\"HF_TOKEN\", \"WANDB_API_KEY\""
    } else {
        "\"HF_TOKEN\""
    };
    let forward_envs = format!("\nforward_envs = [{forward_list}]");
    let forward_envs = forward_envs.as_str();
    let wandb_pip = if wandb_enabled { " wandb" } else { "" };
    let wandb_login = if wandb_enabled {
        r#"
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 \
        && echo "[lora-PLACEHOLDER_NAME] setup: wandb authenticated" \
        || echo "[lora-PLACEHOLDER_NAME] setup: wandb login failed (continuing without W&B)" >&2
else
    echo "[lora-PLACEHOLDER_NAME] setup: WANDB_API_KEY not set; W&B disabled" >&2
fi
"#
    } else {
        ""
    };
    // The wandb_login fragment uses PLACEHOLDER_NAME so the
    // double-curly format escape doesn't conflict with the inner
    // bash. We do the name interpolation as a separate `replace`
    // *after* format!() has resolved its own curly placeholders.
    let wandb_login = wandb_login.replace("PLACEHOLDER_NAME", &a.name);
    let metric_logger_yaml = if wandb_enabled {
        format!(
            "  _component_: torchtune.training.metric_logging.WandBLogger\n  \
             project: {}\n  name: {}",
            a.wandb_project, a.name,
        )
    } else {
        "  _component_: torchtune.training.metric_logging.StdoutLogger".to_string()
    };
    // Network-volume data path. When a volume is configured, the dataset
    // lives on the mounted volume (read at `{mount}/datasets/{name}/<file>`)
    // rather than in the rsync'd workdir, and the heavy `training/full`
    // corpus is excluded from the workdir rsync. An optional `stage` entry
    // uploads the dataset to the volume via S3 before deploy. Top-level
    // keys must precede `[resources]`.
    let vol = a.network_volume_id.trim();
    let (dataset_path, stage_block, volume_block, skip_block) = if !vol.is_empty() {
        let key = format!("datasets/{}", a.name);
        let basename_src = if a.stage_local.is_empty() { &a.dataset_src } else { &a.stage_local };
        let basename = std::path::Path::new(basename_src)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset.jsonl");
        let dataset_path =
            format!("{}/{}/{}", a.volume_mount.trim_end_matches('/'), key, basename);
        let stage_block = if a.stage_local.is_empty() {
            String::new()
        } else {
            format!(
                "\nstage = [{{ local = \"{}\", key = \"{}\" }}]",
                a.stage_local, key
            )
        };
        let mut vb = format!("\nnetwork_volume_id = \"{vol}\"");
        if !a.data_center.trim().is_empty() {
            vb.push_str(&format!("\ndata_center_id = \"{}\"", a.data_center.trim()));
        }
        // Drop the 922 MB private-transcript corpus from the workdir rsync;
        // the dataset is on the volume now, so the workdir only needs code.
        let skip_block = "\nskip_patterns = [\"training/full\"]".to_string();
        (dataset_path, stage_block, vb, skip_block)
    } else {
        (a.dataset_src.clone(), String::new(), String::new(), String::new())
    };
    // Output retrieval. Volume mode is rsync-free: the adapter is written
    // to the mounted volume, tarred into a single key, and `train` pulls
    // that key via S3 (single-object GET — RunPod's S3 list is unusable).
    let mount = a.volume_mount.trim_end_matches('/');
    let (output_dir_expr, tar_block, output_archive_block) = if !vol.is_empty() {
        let archive_key = format!("outputs/adapter-{}.tar.gz", a.name);
        let archive_path = format!("{}/{}", mount, archive_key);
        let output_dir = format!("{}/outputs/adapter-{}", mount, a.name);
        let tar = format!(
            "\necho \"[lora-{name}] train: archiving adapter → {archive}\"\n\
             tar czf {archive} -C {mount}/outputs adapter-{name}",
            name = a.name,
            archive = archive_path,
            mount = mount,
        );
        let archive_block =
            format!("\noutput_archive = {{ key = \"{archive_key}\", local = \"{archive_key}\" }}");
        (output_dir, tar, archive_block)
    } else {
        (format!("$(pwd)/outputs/adapter-{}", a.name), String::new(), String::new())
    };
    // Top-level keys (name, setup, run) come *before* the `[resources]`
    // table — otherwise TOML parses them as members of that table and
    // runpod-cli's Manifest struct rejects the manifest with
    // `missing field setup`.
    format!(
        r#"name = "lora-{name}"
workdir = "."
# Output path matches what the synthesized run script writes
# to ($(pwd)/outputs/adapter-{name}/). v0.0.23 had this as
# `["adapter-{name}"]` — the rsync pull then looked at the wrong
# path on the pod and silently dropped the adapter.
outputs = ["outputs/adapter-{name}"]{forward_envs}{skip_block}{stage_block}{output_archive_block}
# Detached execution (rules_runpod 0.0.6+): the `run` script is
# launched under setsid on the pod and polled for completion, so a
# mid-training SSH drop no longer kills the run. Training jobs are
# long (hour+) and previously died to `Connection reset by peer` on
# the tethered session; detached is the correct default for them.
detached = true
poll_secs = 30

setup = """
set -euo pipefail
echo "[lora-{name}] setup: installing torchtune + huggingface-cli"

# RunPod's pytorch image ships python3 + pip + CUDA; we just add
# torchtune and the HF hub client.
# Pin torchao + torchtune to versions compatible with torch 2.4
# (the version baked into runpod/pytorch:2.4.0). Bumping either
# unpinned pulls a release expecting torch >= 2.11 (e.g. torch.int1).
pip install --quiet --no-input \
    "torchao==0.5.0" \
    "torchtune==0.4.0" \
    "huggingface_hub[cli]" \
    transformers \
    datasets{wandb_pip}{wandb_login}

# Pre-fetch the base model. `hf download` is idempotent and prints
# the cached path on stdout — capture it for the train step.
echo "[lora-{name}] setup: pre-fetching {base_id}@{base_revision}"
hf download --revision {base_revision} --quiet {base_id} > /tmp/lora-{name}.model_dir
echo "[lora-{name}] setup: model staged at $(cat /tmp/lora-{name}.model_dir)"
"""

run = """
set -euo pipefail
echo "[lora-{name}] train: starting"

MODEL_DIR="$(cat /tmp/lora-{name}.model_dir)"
# Dataset path is baked at build time by the Starlark rule from
# the underlying `lora_dataset`'s source path. v0.0.23 used a
# `find` heuristic that would silently pick the wrong .jsonl (or
# none) when the workspace had multiple — torchtune then ran for
# 0 batches and saved an empty adapter. The explicit path is the
# only correct semantics.
DATASET="{dataset_path}"
if [[ ! -f "${{DATASET}}" ]]; then
    echo "[lora-{name}] train: ERROR — dataset not present at ${{DATASET}}" >&2
    echo "[lora-{name}] train:   (workdir rsync'd? volume mounted at {volume_mount}? $(pwd)?)" >&2
    pwd >&2; ls -la >&2
    exit 2
fi
DATASET_ROWS="$(wc -l < "${{DATASET}}")"
echo "[lora-{name}] train: dataset = ${{DATASET}} (${{DATASET_ROWS}} rows)"
if (( DATASET_ROWS == 0 )); then
    echo "[lora-{name}] train: ERROR — dataset is empty" >&2
    exit 2
fi
OUTPUT_DIR="{output_dir_expr}"
mkdir -p "${{OUTPUT_DIR}}"

# Synthesize a complete torchtune config by injecting the per-job
# paths into a static template. Keeping the recipe attrs at the top
# makes the rendered file diff-friendly when hyperparams change.
cat > /tmp/lora-{name}.yaml <<YAML
# Rendered by runpod_orchestrator write-runpod-manifest at build time.
# Hand-edits get overwritten next bazel run.

output_dir: ${{OUTPUT_DIR}}

tokenizer:
  _component_: {tokenizer}
  path: ${{MODEL_DIR}}/vocab.json
  merges_file: ${{MODEL_DIR}}/merges.txt
  max_seq_len: 2048

model:
  _component_: {model_lora}
  lora_attn_modules: [{target_modules_yaml}]
  apply_lora_to_mlp: False
  lora_rank: {rank}
  lora_alpha: {alpha}
  lora_dropout: 0.0

checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: ${{MODEL_DIR}}
  checkpoint_files:
    - model.safetensors
  output_dir: ${{OUTPUT_DIR}}
  model_type: {ckpt_type}

dataset:
  _component_: torchtune.datasets.chat_dataset
  source: json
  data_files: ${{DATASET}}
  conversation_column: messages
  conversation_style: openai
  packed: false
  train_on_input: false

seed: 0
shuffle: True
batch_size: {micro_batch_size}
gradient_accumulation_steps: {grad_accum_steps}
epochs: {epochs}
max_steps_per_epoch: null
resume_from_checkpoint: False
save_adapter_weights_only: True

optimizer:
  _component_: torch.optim.AdamW
  weight_decay: 0.01
  lr: {learning_rate}
  fused: True

lr_scheduler:
  _component_: torchtune.modules.get_cosine_schedule_with_warmup
  num_warmup_steps: 1

loss:
  _component_: torchtune.modules.loss.CEWithChunkedOutputLoss

device: cuda
dtype: bf16

compile: False
enable_activation_checkpointing: False
metric_logger:
{metric_logger_yaml}
log_every_n_steps: 1
log_peak_memory_stats: True
profiler:
  _component_: torchtune.training.setup_torch_profiler
  enabled: False
YAML

echo "[lora-{name}] train: rendered config:"
cat /tmp/lora-{name}.yaml | sed 's/^/  | /' >&2
echo "[lora-{name}] train: dataset first line:"
head -1 "${{DATASET}}" | head -c 200 >&2
echo >&2
echo "[lora-{name}] train: invoking tune run"
tune run lora_finetune_single_device --config /tmp/lora-{name}.yaml
echo "[lora-{name}] train: complete; outputs at ${{OUTPUT_DIR}}"{tar_block}
"""

[resources]
gpu_type = {gpu_type}
image = "{image}"
cloud_type = "{cloud_type}"{volume_block}
"#,
        name = a.name,
        dataset_path = dataset_path,
        volume_mount = a.volume_mount,
        skip_block = skip_block,
        stage_block = stage_block,
        volume_block = volume_block,
        output_dir_expr = output_dir_expr,
        tar_block = tar_block,
        output_archive_block = output_archive_block,
        forward_envs = forward_envs,
        wandb_pip = wandb_pip,
        wandb_login = wandb_login,
        metric_logger_yaml = metric_logger_yaml,
        gpu_type = gpu_array,
        image = a.image,
        base_id = a.base_id,
        base_revision = a.base_revision,
        tokenizer = fc.tokenizer,
        model_lora = fc.model_lora,
        ckpt_type = fc.checkpoint_model_type,
        target_modules_yaml = target_modules_yaml,
        cloud_type = a.cloud_type,
        rank = a.rank,
        alpha = a.alpha,
        learning_rate = a.learning_rate,
        micro_batch_size = a.micro_batch_size,
        grad_accum_steps = a.grad_accum_steps,
        epochs = a.epochs,
    )
}
