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

## JSON fields

| Field | Required | Meaning |
| --- | --- | --- |
| `global_prompt` | yes | Persistent text with no temporal penalty. |
| `local_prompts` | yes | One local event description per schedule entry. |
| `mode` | no | `gaussian` (default) or `hard`. |
| `sigma` | no | Gaussian falloff, default `0.932`. |
| `segment_lengths` | no | Consecutive internal-frame lengths; must cover the complete timeline. |
| `frame_intervals` | no | Independent half-open `[start, end)` internal-frame intervals. Overlap and gaps are allowed. |

`segment_lengths` and `frame_intervals` are mutually exclusive. If neither is
provided, the implementation uses the same ceil-based consecutive allocation as
the Wan port.

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
- Context-parallel attention is rejected because the local processor requires
  the complete packed request.
- Prompt Relay is installed only for the denoising invocation and the original
  attention processors are restored afterward, including after exceptions.
