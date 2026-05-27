#!/usr/bin/env bash
# local_runner.sh — `bazel run :<name>.run` entry point when
# `lora_train(backend = "local")`. Drives torchtune on the host
# machine; no pod, no rsync.
#
# Args (injected by the macro via $(rlocationpath ...)):
#   $1 — runfiles-relative path to the rendered torchtune recipe YAML
#   $2 — runfiles-relative path to the validated SFT JSONL
#   $3 — adapter name (e.g. `parser_jobspec`)
#   $4 — HF hub repo id (e.g. `Qwen/Qwen2.5-1.5B-Instruct`)
#   $5 — HF hub revision (sha or branch)
#   $6 — torchtune family (`qwen2`, `llama3`, `mistral`)
#   $7 — LoRA rank
#   $8 — LoRA alpha
#   $9 — comma-separated target modules
#   ${10} — learning rate
#   ${11} — micro batch size
#   ${12} — gradient accumulation steps
#   ${13} — epochs
#
# Behavior:
#   * Use Bazel's bash runfiles helper to resolve the runfiles-relative
#     paths to absolute on-disk paths.
#   * `cd` to $BUILD_WORKSPACE_DIRECTORY so outputs land in the user's
#     workspace (`outputs/adapter-<name>/`), not the runfiles tree.
#   * Set up a workspace-local Python venv at `.venvs/lora-local/` and
#     install torchao 0.5.0 + torchtune 0.3.1 (same versions as the
#     RunPod backend; consistent training behavior across backends).
#   * Pre-fetch the base model into the user's HF cache.
#   * Render an effective torchtune config inline, with `device: mps`
#     on macOS and `device: cuda` on Linux (if nvidia-smi is present).
#   * Invoke `tune run lora_finetune_single_device --config ...`.

set -uo pipefail

# --- begin runfiles.bash initialization v3 ---
set +e
f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(dirname "$0").runfiles/$f" 2>/dev/null || \
  source "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")").runfiles/$f" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f (runfiles library missing)"; exit 1; }
f=
set -e
# --- end runfiles.bash initialization v3 ---

if [[ $# -lt 13 ]]; then
    echo "fatal: local_runner.sh expects 13 positional args; got $#" >&2
    exit 2
fi

RECIPE="$(rlocation "$1")"
DATASET="$(rlocation "$2")"
NAME="$3"
BASE_ID="$4"
BASE_REV="$5"
FAMILY="$6"
RANK="$7"
ALPHA="$8"
TARGET_MODULES="$9"
LEARNING_RATE="${10}"
MICRO_BATCH_SIZE="${11}"
GRAD_ACCUM_STEPS="${12}"
EPOCHS="${13}"

if [[ -z "$RECIPE"  || ! -f "$RECIPE"  ]]; then echo "fatal: recipe not found"  >&2; exit 3; fi
if [[ -z "$DATASET" || ! -f "$DATASET" ]]; then echo "fatal: dataset not found" >&2; exit 3; fi

cd "${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"

echo "[lora-${NAME}] local: cwd=$PWD adapter=${NAME} family=${FAMILY}"

# ─── Python venv ──────────────────────────────────────────────────
# torchtune's transitive deps churn aggressively; Python 3.14 hits
# `from kagglesdk.kaggle_env import get_web_endpoint` import-time
# breakage. Prefer 3.11 (the ML-stack lingua franca) when available;
# fall back to `python3` otherwise.
PYTHON_BIN="$(command -v python3.11 || command -v python3.12 || command -v python3)"
echo "[lora-${NAME}] local: using $PYTHON_BIN ($($PYTHON_BIN --version))"
VENV=".venvs/lora-local"
if [[ ! -d "$VENV" ]]; then
    echo "[lora-${NAME}] local: creating venv at $VENV"
    "$PYTHON_BIN" -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install once; subsequent runs skip via pip's cache. Let pip pick
# compatible torch / torchao / torchtune versions for the platform —
# the pod backend pins to 2.4 (the image's torch); on macOS / Linux
# locally pip picks the latest matching set.
echo "[lora-${NAME}] local: ensuring torch + torchao + torchtune"
# Known-good pin set for the Apple-Silicon-MPS path. The pip
# dependency graph for the torchtune + torchao + kagglehub triangle
# breaks in interesting ways on every unpinned release; this set is
# the one observed working in May 2026:
#   - torchtune 0.5.0 — has lora_qwen2_1_5b and skips the
#     `int4_weight_only` import path during MPS training.
#   - torchao 0.7.0 — last version that exports `int4_weight_only`
#     (the name torchtune 0.5 references); 0.17 dropped it.
#   - kagglehub < 0.3 — older kagglehub doesn't import
#     `get_web_endpoint` from kagglesdk (the 0.1.x kagglesdk that
#     ships on PyPI doesn't expose it).
#   - torch — let pip pick the latest matching install.
pip install --quiet \
    torch \
    "torchao==0.7.0" \
    "torchtune==0.5.0" \
    "kagglehub<0.3" \
    "huggingface_hub[cli]" \
    transformers \
    datasets >&2

# ─── Device detection (post-install so `import torch` works) ─────
DEVICE="cpu"
if [[ "$(uname)" == "Darwin" ]] && python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
elif command -v nvidia-smi >/dev/null 2>&1; then
    DEVICE="cuda"
fi
echo "[lora-${NAME}] local: device=${DEVICE}"

# ─── Base model fetch ─────────────────────────────────────────────
echo "[lora-${NAME}] local: pre-fetching ${BASE_ID}@${BASE_REV}"
MODEL_DIR="$(hf download --revision "${BASE_REV}" --quiet "${BASE_ID}" 2>&1 | tail -1)"
echo "[lora-${NAME}] local: model staged at ${MODEL_DIR}"

# ─── Render config ────────────────────────────────────────────────
case "${FAMILY}" in
    qwen2)
        TOKENIZER="torchtune.models.qwen2.qwen2_tokenizer"
        MODEL_LORA="torchtune.models.qwen2.lora_qwen2_1_5b"
        CHECKPOINT_MODEL_TYPE="QWEN2"
        ;;
    llama3)
        TOKENIZER="torchtune.models.llama3.llama3_tokenizer"
        MODEL_LORA="torchtune.models.llama3.lora_llama3"
        CHECKPOINT_MODEL_TYPE="LLAMA3"
        ;;
    mistral)
        TOKENIZER="torchtune.models.mistral.mistral_tokenizer"
        MODEL_LORA="torchtune.models.mistral.lora_mistral"
        CHECKPOINT_MODEL_TYPE="MISTRAL"
        ;;
    *)
        echo "fatal: unknown family ${FAMILY}" >&2
        exit 4
        ;;
esac

TARGET_MODULES_YAML=$(echo "${TARGET_MODULES}" | sed 's/,/", "/g')
TARGET_MODULES_YAML='"'${TARGET_MODULES_YAML}'"'

OUTPUT_DIR="$PWD/outputs/adapter-${NAME}"
mkdir -p "${OUTPUT_DIR}"

CONFIG="/tmp/lora-${NAME}-local.yaml"
cat > "${CONFIG}" <<YAML
# Rendered by rules_lora local_runner.sh at run time.
output_dir: ${OUTPUT_DIR}

tokenizer:
  _component_: ${TOKENIZER}
  path: ${MODEL_DIR}/vocab.json
  merges_file: ${MODEL_DIR}/merges.txt
  max_seq_len: 2048

model:
  _component_: ${MODEL_LORA}
  lora_attn_modules: [${TARGET_MODULES_YAML}]
  apply_lora_to_mlp: False
  lora_rank: ${RANK}
  lora_alpha: ${ALPHA}
  lora_dropout: 0.0

checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: ${MODEL_DIR}
  checkpoint_files:
    - model.safetensors
  output_dir: ${OUTPUT_DIR}
  model_type: ${CHECKPOINT_MODEL_TYPE}

dataset:
  _component_: torchtune.datasets.chat_dataset
  source: json
  data_files: ${DATASET}
  conversation_column: messages
  conversation_style: openai
  packed: false
  train_on_input: false

seed: 0
shuffle: True
batch_size: ${MICRO_BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM_STEPS}
epochs: ${EPOCHS}
max_steps_per_epoch: null
resume_from_checkpoint: False
save_adapter_weights_only: True

optimizer:
  _component_: torch.optim.AdamW
  weight_decay: 0.01
  lr: ${LEARNING_RATE}
  fused: False

lr_scheduler:
  _component_: torchtune.modules.get_cosine_schedule_with_warmup
  num_warmup_steps: 1

loss:
  _component_: torchtune.modules.loss.CEWithChunkedOutputLoss

device: ${DEVICE}
dtype: bf16

compile: False
enable_activation_checkpointing: False
metric_logger:
  _component_: torchtune.training.metric_logging.StdoutLogger
log_every_n_steps: 1
log_peak_memory_stats: True
profiler:
  _component_: torchtune.training.setup_torch_profiler
  enabled: False
YAML

echo "[lora-${NAME}] local: invoking tune run on ${DEVICE}"
tune run lora_finetune_single_device --config "${CONFIG}"
echo "[lora-${NAME}] local: complete; outputs at ${OUTPUT_DIR}"
