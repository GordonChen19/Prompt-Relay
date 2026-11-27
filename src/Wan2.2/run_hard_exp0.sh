#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SUITE="${1:-../../evaluation_prompts_truncated.json}"
PROMPT_DIR="prompts/hard_exp0"
OUTPUT_DIR="outputs/hard_exp0"
WAN_PYTHON="${WAN_PYTHON:-$HOME/.conda/envs/LAVA/bin/python}"

mkdir -p "$PROMPT_DIR" "$OUTPUT_DIR"

python3 - "$SUITE" "$PROMPT_DIR" <<'PY'
import json
import sys
from pathlib import Path

suite_path = Path(sys.argv[1])
prompt_dir = Path(sys.argv[2])
entries = json.loads(suite_path.read_text(encoding="utf-8"))

if not isinstance(entries, list) or len(entries) != 20:
    raise ValueError(
        f"Expected a JSON array of 20 prompts, got "
        f"{type(entries).__name__} with {len(entries)} entries"
    )

for original in entries:
    entry = dict(original)
    if len(entry.get("local_prompts", [])) != 3:
        raise ValueError(f"{entry.get('id')}: expected exactly 3 local prompts")
    entry["mode"] = "hard"
    entry["sigma"] = 0.932
    entry["interval_expansion"] = 0
    path = prompt_dir / f"{entry['id']}.json"
    path.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

print(f"Prepared {len(entries)} Wan2.2 prompts with hard masking and expansion=0")
PY

prompt_args=()
for number in $(seq 1 20); do
    printf -v id "p%03d" "$number"
    prompt="$PROMPT_DIR/${id}.json"
    [[ -f "$prompt" ]] || { echo "Missing $prompt" >&2; exit 1; }
    if [[ -s "$OUTPUT_DIR/${id}.mp4" ]]; then
        echo "[skip] $id already exists"
        continue
    fi
    prompt_args+=(--prompt_filepath "$prompt")
done

if (( ${#prompt_args[@]} == 0 )); then
    echo "All Wan2.2 hard masking expansion=0 videos already exist."
    exit 0
fi

CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"$WAN_PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    generate.py \
    --task t2v-A14B \
    --size '832*480' \
    --frame_num 129 \
    --ckpt_dir "$HOME/Wan2.2-T2V-A14B" \
    --prompt_mode relay \
    --query_chunk_size 64 \
    --sample_steps 40 \
    --sample_shift 12 \
    --base_seed 48216 \
    --ulysses_size 2 \
    --dit_fsdp \
    --t5_cpu \
    --offload_model False \
    --output_dir "$OUTPUT_DIR" \
    "${prompt_args[@]}"

echo "Wan2.2 hard masking expansion=0 suite completed: $OUTPUT_DIR"
