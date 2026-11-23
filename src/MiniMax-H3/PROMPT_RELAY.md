# Prompt Relay for MiniMax H3

This integration ports the current Wan2.2 Prompt Relay behavior to the official
MiniMax H3 Diffusers pipeline. It changes no model weights and requires no
training.

H3 uses packed joint attention rather than a dedicated text cross-attention
layer. The integration therefore applies the Wan temporal penalty only to
generated-video-query/local-text-key logits. Global text, audio, keyframes,
references, and all non-video queries retain native attention.

## Run

Install the dependencies from `requirements.txt`, then run from this directory:

```bash
python generate_prompt_relay.py \
  --model_path . \
  --prompt_filepath prompts.json \
  --frame_num 124 \
  --output_path outputs/h3_prompt_relay.mp4
```

Omit `--prompt_filepath` and supply `--prompt` to use the native baseline.

### Multi-GPU

Launch one process per GPU and set `--ulysses_size` to the same value:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 generate_prompt_relay.py \
  --model_path "$HOME/MiniMax-H3" \
  --prompt_filepath prompts.json \
  --frame_num 124 \
  --width 1344 \
  --height 768 \
  --num_inference_steps 50 \
  --cpu_offload true \
  --ulysses_size 4 \
  --output_path outputs/h3_prompt_relay.mp4
```

The packed sequence is divided across ranks. Prompt Relay keeps query rows local
and streams bounded K/V blocks between ranks. Each block's relay penalty is
evaluated in global row coordinates and merged through a float32 online
softmax, so the result is mathematically equivalent to attention over the full
K/V sequence without materializing that sequence on every GPU. After the
distributed denoising loop, only rank 0 loads and runs the video/audio decoders;
the other ranks wait at the final barrier. Only rank 0 writes the video and
metadata. CPU offload reserves 12 GB by default for attention and VAE decoding;
override this with `--memory_reserve_margin` if needed.

This is sequence/context parallelism: it distributes attention activations and
work, but does not shard the 62 GB transformer weights. Each rank still loads a
model replica (CPU offload remains available).

### Native first/last keyframes

The Prompt Relay runner preserves MiniMax-H3's native FL2VA boundary anchors:

```bash
python generate_prompt_relay.py \
  --model_path "$HOME/MiniMax-H3" \
  --prompt_filepath prompts.json \
  --frame_num 124 \
  --image images/start.png \
  --last_image images/end.png \
  --output_path outputs/h3_first_last.mp4
```

`--image` anchors the first generated frame and `--last_image` anchors the
last; either may be supplied alone. Arbitrary intermediate keyframes are not
supported. Encoding, noise augmentation, RoPE placement, and packed-sequence
construction are delegated unchanged to the official FL2VA pipeline. If
`--height` and `--width` are omitted, the native pipeline resolves the canvas
from the boundary image; provide both to request an explicit canvas.

## JSON fields

| Field | Required | Meaning |
| --- | --- | --- |
| `global_prompt` | yes | Persistent text with no temporal penalty. |
| `local_prompts` | yes | One local event description per schedule entry. |
| `mode` | no | `gaussian` (default) or `hard`. |
| `sigma` | no | Gaussian falloff, default `0.932`. |
| `interval_expansion` | no | Expand every resolved interval by this many internal frames on both sides, clamped to the timeline; default `2`. |
| `segment_lengths` | no | Consecutive internal-frame lengths; must cover the complete timeline. |
| `frame_intervals` | no | Independent half-open `[start, end)` internal-frame intervals. Overlap and gaps are allowed. |

`segment_lengths` and `frame_intervals` are mutually exclusive. If neither is
provided, the implementation uses the same ceil-based consecutive allocation as
the Wan port. After resolving those base intervals, an indicated `[x, y)` is
expanded to `[max(0, x-h), min(internal_frames, y+h))`, where `h` is
`interval_expansion`. Set it to `0` to recover the unexpanded schedule.

In Gaussian mode, each interval uses the Wan definitions:

```text
midpoint = (start + end) // 2
window   = (end - start) // 2 - 2
cost     = relu(abs(frame - midpoint) - window)^2 / (2 sigma^2)
```

In hard mode, local text tokens are available only for frames satisfying
`start <= frame < end`. With overlapping intervals, multiple local prompts are
available in the overlap. They still share one softmax, so simultaneous access
does not guarantee successful generation of every event.

The interval indices address generated H3 transformer frames, not decoded
24 FPS output frames. The CLI validates interval bounds after H3 aligns the
requested output length.

## Runtime boundaries

- Supported Diffusers workflows: T2VA, FL2VA, and Ref2VA when enabled through
  `enable_prompt_relay(pipe)`.
- The included CLI exposes T2VA and first/last-frame conditioning.
- Multi-GPU context parallelism is supported through `torchrun` and
  `--ulysses_size`. The custom relay processor uses exact streaming K/V
  attention rather than Diffusers' native Ulysses kernel because the
  frame-dependent additive relay bias is query-specific.
- Prompt Relay is installed only for the denoising invocation and the original
  attention processors are restored afterward, including after exceptions.
