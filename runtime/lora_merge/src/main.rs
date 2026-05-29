//! `lora-merge` — fold a trained LoRA adapter into its base model and
//! export a standalone HF model directory.
//!
//! Folds `W' = W + (alpha/r)·(B @ A)` into the adapter's target
//! projections, casts to bf16, writes `model.safetensors`, and copies
//! the base's `config.json` + tokenizer aux files (`vocab.json`,
//! `merges.txt`, …) so the result loads as a plain causal-LM model
//! with no LoRA at serve/train time. Optionally pushes the merged dir
//! to the HF hub via the `hf` CLI.
//!
//! The merge math is the same as an inference-time LoRA merge (cf.
//! `agora_infer::Engine::export_merged`), but persisted. Runs on CPU —
//! the merge is I/O bound, so no GPU feature is needed.
//!
//! Adapter key layout (torchtune → peft):
//!   `base_model.model.model.layers.{l}.self_attn.{p}.lora_{A,B}.weight`
//! Base key layout (HF Qwen2/Llama):
//!   `model.layers.{l}.self_attn.{p}.weight`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use candle_core::{safetensors as cst, DType, Device, Tensor};
use clap::Parser;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};

#[derive(Parser, Debug)]
#[command(about = "Merge a LoRA adapter into its base model and export an HF dir.")]
struct Cli {
    /// Adapter directory (adapter_model.bin + adapter_config.json).
    #[arg(long)]
    adapter: PathBuf,
    /// HF hub repo id of the base model the adapter was trained on.
    #[arg(long)]
    base_id: String,
    /// HF hub revision (commit sha or branch).
    #[arg(long, default_value = "main")]
    base_revision: String,
    /// Output directory for the merged model.
    #[arg(long)]
    out: PathBuf,
    /// LoRA scale override. Default: alpha/r read from adapter_config.json.
    #[arg(long)]
    lora_scale: Option<f64>,
    /// If set, push the merged dir to this HF repo id via the `hf` CLI.
    #[arg(long)]
    push_repo: Option<String>,
    /// Create the HF repo as private when pushing.
    #[arg(long)]
    private: bool,
}

/// Subset of an HF model `config.json` we need.
#[derive(serde::Deserialize)]
struct ModelConfig {
    num_hidden_layers: usize,
}

/// Subset of a peft/torchtune `adapter_config.json`.
#[derive(serde::Deserialize)]
struct AdapterConfig {
    r: f64,
    lora_alpha: f64,
    target_modules: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let cli = Cli::parse();

    let adapter_bin = cli.adapter.join("adapter_model.bin");
    anyhow::ensure!(
        adapter_bin.exists(),
        "adapter_model.bin not found in {}",
        cli.adapter.display()
    );
    let adapter_cfg: AdapterConfig = serde_json::from_slice(
        &std::fs::read(cli.adapter.join("adapter_config.json"))
            .context("read adapter_config.json")?,
    )
    .context("parse adapter_config.json")?;
    let scale = cli.lora_scale.unwrap_or(adapter_cfg.lora_alpha / adapter_cfg.r);

    // Export is I/O bound; CPU avoids Metal save quirks and keeps the
    // build portable. Merge math in f32 for precision, save in bf16.
    let device = Device::Cpu;

    // ── HF hub: base config + weights ─────────────────────────────
    let api = Api::new().context("init hf-hub api")?;
    let repo = api.repo(Repo::with_revision(
        cli.base_id.clone(),
        RepoType::Model,
        cli.base_revision.clone(),
    ));
    let cfg_path = repo.get("config.json").context("fetch config.json")?;
    let weights_path = repo.get("model.safetensors").context("fetch model.safetensors")?;
    let cfg: ModelConfig =
        serde_json::from_slice(&std::fs::read(&cfg_path).context("read config.json")?)
            .context("parse model config.json")?;

    // ── Base weights → mutable f32 map ─────────────────────────────
    let base = cst::load(&weights_path, &device).context("load base safetensors")?;
    let mut weights: HashMap<String, Tensor> = HashMap::with_capacity(base.len());
    for (k, t) in base {
        weights.insert(k, t.to_dtype(DType::F32).context("cast base tensor")?);
    }

    // ── Merge (f32) ────────────────────────────────────────────────
    merge_lora(
        &mut weights,
        &adapter_bin,
        cfg.num_hidden_layers,
        &adapter_cfg.target_modules,
        scale,
        &device,
    )?;

    // Cast merged weights to bf16 for the saved checkpoint.
    for v in weights.values_mut() {
        *v = v.to_dtype(DType::BF16).context("cast merged weight")?;
    }

    // ── Write merged dir ───────────────────────────────────────────
    std::fs::create_dir_all(&cli.out)
        .with_context(|| format!("create {}", cli.out.display()))?;
    let st_path = cli.out.join("model.safetensors");
    cst::save(&weights, &st_path)
        .with_context(|| format!("save {}", st_path.display()))?;
    tracing::info!(tensors = weights.len(), path = %st_path.display(), "merged weights saved");

    // Copy the aux files a standalone load needs. config.json +
    // tokenizer.json are mandatory; the rest are best-effort (present
    // for Qwen2.5). vocab.json + merges.txt are required by torchtune's
    // qwen2_tokenizer, so copy them when the base ships them.
    for fname in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
    ] {
        if let Ok(src) = repo.get(fname) {
            std::fs::copy(&src, cli.out.join(fname))
                .with_context(|| format!("copy {fname}"))?;
        }
    }
    tracing::info!(out_dir = %cli.out.display(), "merged HF model exported");

    // ── Optional HF push ───────────────────────────────────────────
    if let Some(repo_id) = &cli.push_repo {
        push_to_hub(repo_id, &cli.out, cli.private)?;
    }

    Ok(())
}

/// Merge `W + scale·(B @ A)` into the target projection weights for
/// every layer. `target_modules` names the projections to merge
/// (e.g. `q_proj`, `k_proj`, `v_proj`, `o_proj`) — all under
/// `self_attn`. Non-attention modules (mlp) are skipped with a warning.
fn merge_lora(
    weights: &mut HashMap<String, Tensor>,
    adapter_bin: &Path,
    num_layers: usize,
    target_modules: &[String],
    scale: f64,
    device: &Device,
) -> Result<()> {
    let pairs = candle_core::pickle::read_all(adapter_bin)
        .context("read LoRA .bin (candle pickle)")?;
    let lora: HashMap<String, Tensor> = pairs.into_iter().collect();

    let attn = ["q_proj", "k_proj", "v_proj", "o_proj"];
    let mut merged = 0usize;
    for l in 0..num_layers {
        for p in target_modules {
            if !attn.contains(&p.as_str()) {
                tracing::warn!(module = %p, "non-attention LoRA module not supported yet — skipped");
                continue;
            }
            let base_key = format!("model.layers.{l}.self_attn.{p}.weight");
            let pre = format!("base_model.model.model.layers.{l}.self_attn.{p}");
            let a_key = format!("{pre}.lora_A.weight");
            let b_key = format!("{pre}.lora_B.weight");
            let (Some(a), Some(b)) = (lora.get(&a_key), lora.get(&b_key)) else {
                continue;
            };
            let a = a.to_dtype(DType::F32)?.to_device(device)?;
            let b = b.to_dtype(DType::F32)?.to_device(device)?;
            // (out,r) @ (r,in) = (out,in), same layout as base W.
            let delta = (b.matmul(&a)? * scale)?;
            let base_w = weights
                .get(&base_key)
                .with_context(|| format!("base weight missing: {base_key}"))?;
            let new_w = base_w
                .broadcast_add(&delta)
                .with_context(|| format!("merge delta into {base_key}"))?;
            weights.insert(base_key, new_w);
            merged += 1;
        }
    }
    anyhow::ensure!(
        merged > 0,
        "LoRA merge applied to 0 weights — adapter key prefix mismatch? expected \
         `base_model.model.model.layers.N.self_attn.{{q,k,v,o}}_proj.lora_{{A,B}}.weight`"
    );
    tracing::info!(merged_projections = merged, scale, "LoRA merged");
    Ok(())
}

/// Push the merged dir to the HF hub via the `hf` CLI (symmetric with
/// the on-pod `hf download`). Pre-creates the repo so `--private` is
/// honored (plain `hf upload` would create it public).
fn push_to_hub(repo_id: &str, dir: &Path, private: bool) -> Result<()> {
    if Command::new("hf").arg("version").output().is_err() {
        bail!(
            "`hf` CLI not found on PATH — needed for --push-repo. \
             Install `huggingface_hub[cli]` or drop --push-repo and push manually."
        );
    }
    tracing::info!(repo = repo_id, private, "creating HF repo (idempotent)");
    let mut create = Command::new("hf");
    create.args(["repo", "create", repo_id, "--repo-type", "model", "-y"]);
    if private {
        create.arg("--private");
    }
    // exist_ok: a non-zero here usually just means "already exists".
    let _ = create.status();

    tracing::info!(repo = repo_id, dir = %dir.display(), "uploading to HF");
    let status = Command::new("hf")
        .args(["upload", repo_id])
        .arg(dir)
        .args([".", "--repo-type", "model"])
        .status()
        .context("spawn `hf upload`")?;
    if !status.success() {
        bail!("`hf upload` exited with status {status}");
    }
    tracing::info!(repo = repo_id, "push complete");
    Ok(())
}
