# VBench evaluation for Prompt Relay and Visual Relay

This folder does two things:

1. **Make the videos.** For every prompt in a prompt file, it runs our method and the plain model it is built on.
2. **Score the videos** with VBench and put the methods side by side in one table.

Follow the steps in order. Run every command from the repo root (`Prompt-Relay/`).
Details (how videos are split, every option, every output column) are in [REFERENCE.md](REFERENCE.md).

---

## Part A: one-time setup

### A1. Get the code

```bash
git clone --recurse-submodules -b vbench-evaluation https://github.com/GordonChen19/Prompt-Relay.git
cd Prompt-Relay
```

This also fetches the Wan 2.2 code (`src/Wan2.2`), the Visual Relay code (`src/Visual-Relay`) and VBench
(`vbench_eval/third_party/VBench`). The MiniMax H3 code is in `src/MiniMax-H3`.

### A2. Make the Wan environment (skip if you already run Wan 2.2)

```bash
python3.10 -m venv ~/venv && source ~/venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r src/Wan2.2/requirements.txt
```

If `flash_attn` fails to build, install everything else first, then `pip install flash-attn --no-build-isolation`.

### A3. Make the VBench environment

VBench needs older packages than Wan, so it gets its own environment.

```bash
python3.10 -m venv ~/venvs/vbench && source ~/venvs/vbench/bin/activate
pip install -U pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r vbench_eval/third_party/VBench/requirements.txt -r vbench_eval/requirements.txt
```

### A4. Download the model weights

Each is about 126 GB. Prompt Relay needs T2V; Visual Relay needs I2V.

```bash
source ~/venv/bin/activate
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B
mkdir -p ~/checkpoints
ln -sfn ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B/snapshots/* ~/checkpoints/Wan2.2-T2V-A14B
ln -sfn ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/* ~/checkpoints/Wan2.2-I2V-A14B
```

If a download stops, run the same command again; it continues where it stopped.

### A5. Check the paths in `vbench_eval/methods.json`

Open the file and look at `"paths"`. The defaults are:

| Key | Default | What |
|---|---|---|
| `wan_python` | `~/venv/bin/python` | the Wan environment from A2 |
| `wan_t2v_ckpt` | `~/checkpoints/Wan2.2-T2V-A14B` | T2V weights from A4 |
| `wan_i2v_ckpt` | `~/checkpoints/Wan2.2-I2V-A14B` | I2V weights from A4 |

Change them only if your folders are elsewhere.

### A6. Check that everything is in place (no GPU needed)

```bash
source ~/venvs/vbench/bin/activate
python vbench_eval/check_env.py
python vbench_eval/tests/test_offline.py
```

**Expect:** `check_env.py` ends with `0 failure(s)`; the tests end with `... checks passed` and no `[FAIL]` line.

---

## Part B: test with one video

### B1. Make the video

```bash
source ~/venv/bin/activate
python vbench_eval/generate_suite.py --prompts vbench_eval/examples/single_prompt_test.json \
    --methods wan2.2_t2v --out_root outputs/test --gpus 0,1
```

**Expect:** `[generate] 1/1 wan2.2_t2v / dog_park: done in ... min`, and the video at
`outputs/test/wan2.2_t2v/dog_park.mp4`. If it fails, the error is at the end of
`outputs/test/wan2.2_t2v/logs/dog_park.log`.

### B2. Score it

```bash
source ~/venvs/vbench/bin/activate
CUDA_VISIBLE_DEVICES=0 python vbench_eval/run_all.py --prompts vbench_eval/examples/single_prompt_test.json \
    --root outputs/test --out eval_results/test
```

The first time, this downloads the scoring models (a few GB).

### B3. Read the result

Open `eval_results/test/comparison.md`: one row per method, one column per score.
`eval_results/test/wan2.2_t2v/run_info.json` shows how many seconds each score took.

---

## Part C: run the examples

### C1. Prompt Relay (text prompts)

```bash
source ~/venv/bin/activate
python vbench_eval/generate_suite.py --prompts vbench_eval/examples/prompt_relay_prompts.json \
    --out_root outputs/prompt_relay --gpus 0,1
source ~/venvs/vbench/bin/activate
CUDA_VISIBLE_DEVICES=0 python vbench_eval/run_all.py --prompts vbench_eval/examples/prompt_relay_prompts.json \
    --root outputs/prompt_relay --out eval_results/prompt_relay
```

For each prompt this makes two videos: **plain Wan 2.2** (`wan2.2_t2v`) and **Wan 2.2 with Prompt Relay**
(`wan2.2_t2v_prompt_relay`). You do not write a prompt for the plain model: it gets the global and local
prompts joined into one.

### C2. Visual Relay (two images per prompt)

```bash
source ~/venv/bin/activate
python vbench_eval/generate_suite.py --prompts vbench_eval/examples/visual_relay_prompts.json \
    --out_root outputs/visual_relay --gpus 0,1
source ~/venvs/vbench/bin/activate
CUDA_VISIBLE_DEVICES=0 python vbench_eval/run_all.py --prompts vbench_eval/examples/visual_relay_prompts.json \
    --root outputs/visual_relay --out eval_results/visual_relay
```

For each prompt this makes **plain Wan 2.2 I2V** (`wan2.2_i2v`: starts from image 1, prompts joined) and
**Wan 2.2 I2V with Visual Relay** (`wan2.2_i2v_visual_relay`: image 1 at the start, image 2 later).

Tip: add `--print_only` to any `generate_suite.py` command to see the exact commands without running them.

---

## Part D: your own prompts

### D1. Write a prompt file

Copy an example and replace its entries with yours (one entry per video, format under "Prompt files" below):

```bash
cp vbench_eval/examples/prompt_relay_prompts.json my_text_prompts.json      # text prompts
cp vbench_eval/examples/visual_relay_prompts.json my_image_prompts.json     # prompts with two images
```

Keep text prompts and image prompts in separate files, so each gets its own folders and its own results table.

### D2. Pick two folder names for this prompt file

| Option | What it is | Example |
|---|---|---|
| `--out_root` | where the videos go | `outputs/my_text_prompts` |
| `--out` | where the scores go | `eval_results/my_text_prompts` |

Use new names for each prompt file, so its videos and scores are not mixed with another file's.

### D3. Make the videos, then score them

```bash
source ~/venv/bin/activate
python vbench_eval/generate_suite.py --prompts my_text_prompts.json \
    --out_root outputs/my_text_prompts --gpus 0,1
source ~/venvs/vbench/bin/activate
CUDA_VISIBLE_DEVICES=0 python vbench_eval/run_all.py --prompts my_text_prompts.json \
    --root outputs/my_text_prompts --out eval_results/my_text_prompts
```

The results table is `eval_results/my_text_prompts/comparison.md`.

### D4. Stopping, restarting and adding prompts

Each video is saved as `<out_root>/<method>/<file_name>`, for example
`outputs/my_text_prompts/wan2.2_t2v_prompt_relay/chef_chop_then_fry.mp4`. Before making a video,
`generate_suite.py` checks whether that file already exists; if it does, it skips it. So you can:

* **stop and restart:** run the same command again; only the missing videos are made;
* **add prompts:** add entries to the file and run the same command again; only the new ones are made.

Scoring works the same way: `run_all.py` keeps the scores of videos it has already scored and scores only new
or changed ones.

To make or score everything again, add `--force` to the command. To redo a single video, delete its `.mp4`
and run the command again.

---

## Prompt files

A text prompt (Prompt Relay):

```json
{
  "id": "chef_chop_then_fry",
  "file_name": "chef_chop_then_fry.mp4",
  "global_prompt": "A chef in a white uniform at a wooden counter in a bright kitchen, fixed camera.",
  "local_prompts": [
    "The chef chops a red bell pepper into thin strips with a large knife.",
    "The chef tosses the pepper strips into a hot pan on the stove, and steam rises."
  ]
}
```

An image prompt (Visual Relay) adds two images and where image 2 goes:

```json
{
  "id": "rock_feather_fall",
  "file_name": "rock_feather_fall.mp4",
  "global_prompt": "A rock and a white feather above the ground in an outdoor scene. ...",
  "local_prompts": ["The rock and the white feather fall ...", "The rock lies still on the ground ..."],
  "images": ["src/Visual-Relay/fnr1.png", "src/Visual-Relay/fnr2.png"],
  "image_latents": [0, 10]
}
```

| Field | What to put |
|---|---|
| `id` | a short name for the prompt (letters, digits, `_`) |
| `file_name` | `<id>.mp4`; every method saves its video under this name in its own folder |
| `global_prompt` | what is true for the whole video (who, where, camera); no timing words |
| `local_prompts` | the events, in order. The video is split evenly between them: 2 prompts = 2 halves |
| `images` | image prompts only: the first and the second image |
| `image_latents` | image prompts only: `[0, N]` puts image 2 at latent N of 21 (video frame 4N-3); `[0, 14]` is about 2/3 of the way in. The video is split there instead of in halves |

A file is a list of entries: `[ {...}, {...} ]`.

---

## Methods

| Method | Runs on | What it is |
|---|---|---|
| `wan2.2_t2v` | text prompts | plain Wan 2.2 T2V-A14B, prompts joined into one |
| `wan2.2_t2v_prompt_relay` | text prompts | Wan 2.2 T2V-A14B with Prompt Relay |
| `wan2.2_i2v` | image prompts | plain Wan 2.2 I2V-A14B from image 1, prompts joined into one |
| `wan2.2_i2v_visual_relay` | image prompts (2 images, 2 local prompts) | Wan 2.2 I2V-A14B with Visual Relay |
| `minimax_h3`, `minimax_h3_prompt_relay` | text prompts | **switched off** until set up: check the arguments in `methods.json`, then set `"enabled": true` |

All Wan methods use the same settings (`methods.json`, `"settings"`): seed 48216, 832x480, 81 frames
(about 5 s at 16 fps); the two I2V methods use sample shift 3. `--gpus 0,1` runs each video on 2 GPUs;
`--gpus 0` on one (slower, with offloading).

---

## What is measured

Each video is split into one segment per local prompt. Each segment is scored against its own prompt, and the
moment where one prompt hands over to the next (the "switch") is scored on its own. See REFERENCE.md.

**Used, from VBench:**

| Score | What it tells us |
|---|---|
| overall_consistency | does each segment match its own prompt (and the whole video the global prompt) |
| subject_consistency | does the main subject keep its identity, within segments and across the switch |
| background_consistency | does the scene stay the same |
| temporal_flickering | frame-to-frame flicker |
| motion_smoothness | is the motion smooth |
| dynamic_degree | is there motion at all (a still video would win the scores above) |
| aesthetic_quality | how good the frames look |
| imaging_quality | blur, noise, over-exposure |

**Used, from VBench++ (image prompts only):** i2v_subject and i2v_background: how closely segment k follows
image k.

**Not used:** object class, multiple objects, color, spatial relationship, scene, human action (they need
VBench's own labelled prompts or action list); appearance style and temporal style (not our question); camera
motion (scores one fixed camera move per video only); fairness, bias and safety (need VBench's own prompt
sets). VBench-2.0 is not used.

**Extra scores (not VBench), each can be switched off:**

| Score | Switch off with | What |
|---|---|---|
| clip_t | `--no_clip_t` | CLIP text-frame match of each segment with its own prompt |
| prompt_match | `--no_prompt_match` | share of segments whose own prompt matches best (are the events in order?) |
| clip_f | `--no_clip_f` | similarity of neighbouring frames; lowest value around the switch |
| gen_time | `--no_gen_time` | seconds to make the video (model loading included) |

Add the switch to the `run_all.py` command, e.g. `... run_all.py ... --no_clip_t --no_clip_f`.

---

## Results

`eval_results/<name>/comparison.md` (and `.csv`): one row per method, the mean of each score, and how many
videos it covers. Higher is better, except `gen_time_s`.

Per method, `eval_results/<name>/<method>/` has `per_video.csv` (one row per video) and `summary.csv`
(mean, standard deviation, count). The most useful columns:

| Column | Meaning |
|---|---|
| `overall_consistency_seg_mean`, `clip_t_seg_mean` | how well the segments match their own prompts |
| `prompt_match_acc` | share of segments in the right order |
| `subject_consistency_switch_min` | identity kept through the switch |
| `i2v_subject_img2` | how closely the video after the switch follows image 2 |
