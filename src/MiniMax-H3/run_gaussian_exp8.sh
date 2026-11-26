#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SUITE="${1:-../../evaluation_prompts.json}"
PROMPT_DIR="prompts/gaussian_exp8"
OUTPUT_DIR="outputs/gaussian_exp8"
LOG_DIR="logs/gaussian_exp8"
H3_PYTHON="${H3_PYTHON:-$HOME/.conda/envs/minimax-h3/bin/python}"

mkdir -p "$PROMPT_DIR" "$OUTPUT_DIR" "$LOG_DIR"

python3 - "$SUITE" "$PROMPT_DIR" <<'PY'
import json
import sys
from pathlib import Path

suite_path = Path(sys.argv[1])
prompt_dir = Path(sys.argv[2])
entries = json.loads(suite_path.read_text(encoding="utf-8"))

if not isinstance(entries, list) or len(entries) != 20:
    raise ValueError(f"Expected a JSON array of 20 prompts, got {type(entries).__name__} with {len(entries)} entries")

for original in entries:
    entry = dict(original)
    if not entry.get("local_prompts"):
        raise ValueError(f"{entry.get('id')}: local_prompts is empty")
    entry["mode"] = "gaussian"
    entry["sigma"] = 0.932
    entry["interval_expansion"] = 8
    path = prompt_dir / f"{entry['id']}.json"
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"Prepared {len(entries)} MiniMax-H3 prompts with Gaussian expansion=8")
PY

for number in $(seq 1 20); do
    printf -v id "p%03d" "$number"
    prompt="$PROMPT_DIR/${id}.json"
    output="$OUTPUT_DIR/${id}.mp4"

    if [[ -s "$output" ]]; then
        echo "[skip] $id already exists"
        continue
    fi

    echo "[generate] $id: Gaussian sigma=0.932, interval expansion=8"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$H3_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        generate_prompt_relay.py \
        --model_path "$HOME/MiniMax-H3" \
        --prompt_filepath "$prompt" \
        --frame_num 345 \
        --width 832 \
        --height 480 \
        --num_inference_steps 50 \
        --seed 48216 \
        --query_chunk_size 128 \
        --cpu_offload true \
        --memory_reserve_margin 12GB \
        --ulysses_size 4 \
        --output_path "$output" \
        2>&1 | tee "$LOG_DIR/${id}.log"

    echo "[done] $id"
done

echo "MiniMax-H3 Gaussian expansion=8 suite completed: $OUTPUT_DIR"
