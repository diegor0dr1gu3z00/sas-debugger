#!/usr/bin/env bash
# Serve the trained adapter on a llama.cpp LLM server (llama-server).
#
#   1. Build a GGUF for the base model (Qwen/Qwen2.5-0.5B-Instruct) if missing.
#   2. Merge the LoRA adapter (Peft safetensors) into it if an adapter is given.
#   3. (Optionally) quantize to Q4_K_M.
#   4. Start llama-server on the OpenAI-compatible endpoint.
#
# Then benchmark it against the transformers backend:
#   "$(venv_py)" -m db_debug_rl.evaluate_lm --backend llama.cpp \
#       --llama_url http://127.0.0.1:8080 \
#       --test_jsonl data/generated/external_episodes.jsonl \
#       --json_out eval/external-llama.json --tag llama-ext
#
# Env:
#   LLAMA_DIR   llama.cpp checkout (default: ~/llama.cpp)
#   ADAPTER     Peft adapter dir to merge (default: checkpoints/sft-0.5b-ext/lora)
#   BASE_MODEL  HF base model (default: Qwen/Qwen2.5-0.5B-Instruct)
#   BUILD       llama.cpp build dir (default: $LLAMA_DIR/build)
#   QUANT       quantize/reuse Q4_K_M in addition to f16 (default: 1)
#   PORT        llama-server port (default: 8080)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
. scripts/_venv.sh

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
BUILD="${BUILD:-$LLAMA_DIR/build}"
BIN="$BUILD/bin"
ADAPTER="${ADAPTER:-checkpoints/sft-0.5b-ext/lora}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8080}"

[[ -d "$LLAMA_DIR" ]] || { echo "ERROR: $LLAMA_DIR not found. Clone llama.cpp:" >&2
  echo "  git clone https://github.com/ggml-org/llama.cpp $LLAMA_DIR && cmake -B $BUILD -DGGML_CUDA=ON && cmake --build $BUILD -j" >&2
  exit 1; }
for t in convert_hf_to_gguf.py llama-export-lora llama-server; do
  [[ -e "$LLAMA_DIR/$t" || -e "$BIN/$t" ]] || { echo "ERROR: missing $t — build llama.cpp ($BUILD)." >&2; exit 1; }
done

PY="${PY:-$(venv_py)}"

# 1. Base GGUF (f16).
BASE_F16="checkpoints/${BASE_MODEL##*/}-f16.gguf"
if [[ ! -e "$BASE_F16" ]]; then
  echo "convert $BASE_MODEL -> $BASE_F16"
  "$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$BASE_MODEL" \
    --outfile "$BASE_F16" --outtype f16
fi

# 2. Merge LoRA.
MERGED_F16="checkpoints/${BASE_MODEL##*/}-${ADAPTER##*/}-f16.gguf"
if [[ ! -e "$MERGED_F16" ]]; then
  echo "merge LoRA ($ADAPTER) -> $MERGED_F16"
  "$BIN/llama-export-lora" -m "$BASE_F16" --lora "$ADAPTER" \
    -o "$MERGED_F16"
fi

# 3. Quantize.
if [[ "${QUANT:-1}" == "1" ]]; then
  MERGED="$MERGED_F16"
else
  MERGED="${MERGED_F16%-f16.gguf}-q4_k_m.gguf"
  [[ -e "$MERGED" ]] || { echo "quantize -> $MERGED"; "$BIN/llama-quantize" "$MERGED_F16" "$MERGED" Q4_K_M; }
fi

# 4. Serve.
echo "llama-server on http://127.0.0.1:${PORT} (model: $MERGED)"
echo "Then run the eval with: \"$(venv_py)\" -m db_debug_rl.evaluate_lm --backend llama.cpp --llama_url http://127.0.0.1:${PORT} --test_jsonl data/generated/external_episodes.jsonl --tag llama"
exec "$BIN/llama-server" -m "$MERGED" --host 127.0.0.1 --port "$PORT" -ngl 99
