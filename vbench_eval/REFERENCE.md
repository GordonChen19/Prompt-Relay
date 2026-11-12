# vbench_eval reference

The step-by-step guide is [README.md](README.md). This file has the details.

- [Files](#files)
- [Environments](#environments)
- [Model weights](#model-weights)
- [Making the videos: generate_suite.py](#making-the-videos-generate_suitepy)
- [Prompt file: every field](#prompt-file-every-field)
- [How a video is split](#how-a-video-is-split)
- [Metrics: used and not used](#metrics-used-and-not-used)
- [Scoring options: evaluate.py and run_all.py](#scoring-options-evaluatepy-and-run_allpy)
- [Output files and columns](#output-files-and-columns)
- [Cautions](#cautions)

## Files

```
vbench_eval/
  generate_suite.py    prompt file -> every method's videos in OUT_ROOT/<method>/<file_name>
  methods.json         each method's code folder, environment, weights and arguments; paths and settings
  run_all.py           scores every method folder under one directory, writes comparison.csv / .md
  evaluate.py          scores one folder of videos
  timed_run.py         wraps any generation command; records its time (used by generate_suite.py)
  check_env.py         checks the VBench environment (versions, VBench modules, GPU)
  requirements.txt     packages added to VBench's own, with pins that keep them compatible
  vbeval/              suite.py (prompt file), segments.py (splitting), video_io.py, metrics_vbench.py,
                       metrics_extra.py, report.py
  examples/            single_prompt_test.json, prompt_relay_prompts.json, visual_relay_prompts.json
  third_party/VBench/  VBench, git submodule pinned to commit fd18b3d
  tests/test_offline.py     checks without GPU or models
```

## Environments

| Environment | Used for | Key versions |
|---|---|---|
| `~/venv` | Wan 2.2 (plain, Prompt Relay, Visual Relay) and `generate_suite.py` | torch 2.5.1+cu121, transformers 4.49-4.51.3, numpy < 2, flash_attn (`src/Wan2.2/requirements.txt`) |
| `~/venvs/h3` | MiniMax H3 | torch >= 2.6, transformers 5.17.0, diffusers at a pinned commit (`src/MiniMax-H3/requirements.txt`) |
| `~/venvs/vbench` | scoring | transformers 4.33.2, numpy < 2, pyiqa 0.1.10 |

They cannot be merged: VBench pins transformers 4.33.2, Wan needs 4.49-4.51.3, MiniMax H3 needs 5.17.0.
`requirements.txt` pins `pyiqa==0.1.10` because every later release needs a newer transformers than VBench's.
VBench is used as is: this tool imports VBench's own code and calls its scoring functions. A different VBench
clone can be used with `--vbench_root` or `VBENCH_ROOT`.

Scoring models download on the first scoring run (DINO, CLIP, ViCLIP, RAFT, AMT, MUSIQ, LAION aesthetic,
DreamSim) into `~/.cache/vbench`, `~/.cache/clip` and `~/.cache/torch`. DreamSim saves its weights to
`./models` under the folder you run from (VBench's call); `models/` is git-ignored.

## Model weights

Each run loads the weights from a folder on disk. `methods.json` (`"paths"`) points to them:

| Key | Model |
|---|---|
| `wan_t2v_ckpt` | Wan2.2-T2V-A14B (plain T2V, Prompt Relay) |
| `wan_i2v_ckpt` | Wan2.2-I2V-A14B (plain I2V, Visual Relay) |
| `h3_model` | MiniMax H3: a Hugging Face name (`MiniMaxAI/MiniMax-H3`, downloaded on first use) or a folder |

T2V and I2V share two files, the T5 text encoder `models_t5_umt5-xxl-enc-bf16.pth` (about 11 GB) and the VAE
`Wan2.1_VAE.pth`. To keep one copy, download one model fully, download the other with
`--exclude "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.1_VAE.pth"` and link the two files into its folder.

## Making the videos: generate_suite.py

```
python vbench_eval/generate_suite.py --prompts FILE.json [--out_root outputs] [--gpus 0,1]
       [--methods a,b] [--ids id1,id2] [--print_only] [--force] [--config vbench_eval/methods.json]
```

| Option | Default | Meaning |
|---|---|---|
| `--prompts` | required | the prompt file |
| `--out_root` | `outputs` | videos go to `OUT_ROOT/<method>/<file_name>` |
| `--gpus` | `$CUDA_VISIBLE_DEVICES`, else `0,1` | GPUs for each video. Wan: 1, 2, 4 or 8 GPUs; 2 or more run torchrun with FSDP and Ulysses, 1 uses offloading |
| `--methods` | every enabled method | only these methods |
| `--ids` | all | only these prompts |
| `--print_only` | | print each command; run and write nothing |
| `--force` | | make videos again even if they exist |
| `--master_port` | 29500 + first GPU id | torchrun port; differs per GPU set so two runs can share a machine |

Which methods run on which prompts: text-only methods on prompts without images, `wan2.2_i2v` on prompts
with images, `wan2.2_i2v_visual_relay` on prompts with exactly 2 images and 2 local prompts. Methods with
`"enabled": false` in `methods.json` are skipped.

For each video it writes `<name>.meta.json` beside it: `gen_time_s` (seconds, model loading included) and, for
image prompts, `switch_frames` (the frame where image 2 starts, 4N-3 for `image_latents` `[0, N]`), and a log
in `OUT_ROOT/<method>/logs/<id>.log`. Prompt Relay's input file goes to `OUT_ROOT/_prompt_files/<id>.json`.
One failed video does not stop the rest; the run ends with exit code 1 if any failed.

Adding a method: add an entry to `methods.json` with `runs_on`, `python`, `code_dir`, `script`, `parallel`
and `args`. Placeholders in `args`: any `paths` or `settings` key, and per prompt `{output}`,
`{prompt_joined}`, `{prompt_file}`, `{global_prompt}`, `{local_1}`, `{local_2}`, ..., `{image_1}`,
`{image_2}`, ..., `{image_latent_2}`, ...

A video made some other way can be timed with
`python vbench_eval/timed_run.py --video OUT.mp4 [--switch_latents N] -- <command>`.

## Prompt file: every field

| Field | Required | Meaning |
|---|---|---|
| `file_name` | yes | video file name, `.mp4`, no folders; the same in every method's folder |
| `id` | no | defaults to `file_name` without `.mp4`; must be unique |
| `global_prompt` | this or `local_prompts` | what holds for the whole video; no timing words |
| `local_prompts` | no | the events in order; 2 or more = the video is split into that many segments |
| `images` | no | image k is compared with segment k (VBench++ I2V scores) |
| `image_latents` | for Visual Relay | latent of each image, first one 0; `[0, 14]` = image 2 at latent 14 of 21 |
| `file_dir` | no | folder of the video, only for `evaluate.py` without `--videos_dir` |
| `switch_fractions` | no, rarely | where each prompt switch is, as a fraction of the video's length, for a video made elsewhere whose switch is not at the even split. `[0.688]` = 68.8% in |
| anything else | no | kept and copied to the results as `extra` (e.g. `"tags": ["intended_transition"]`) |

A file is one entry, a list of entries, or `{"prompts": [...]}`.

## How a video is split

VBench scores whole files and knows nothing about prompts that change over time or about image 2. So this
tool splits each video into short clips, gives each clip to VBench as its own "video" (with its own prompt
or image), and maps the scores back.

**Where the prompt switches** (first frame of segment 2, 3, ...), first match wins:
1. `switch_frames` in the video's `.meta.json` (exact; written for image prompts);
2. the entry's `switch_fractions`;
3. an even split: segment k starts at `round(k * frames / N)` (a 9-second video with 3 prompts: 3 s each).

**Example:** a Visual Relay video, 77 frames at 16 fps, image 2 from frame 53, default margins of 0.5 s (8 frames):

```
frame:   0 ................................ 44 45 ...... 52|53 ...... 60 61 ............ 76
segment: |----------------- segment 1 ----------------------|---------- segment 2 ---------|
core:    |------------- core 1 --------------|                           |---- core 2 -----|
switch:                                      |------ switch window ------|
```

* **segment k**: from its switch to the next.
* **core k**: segment k without `margin_s` next to a switch. Prompt Relay hands over gradually, so the frames
  next to a switch belong to neither prompt and are left out of the prompt-matching scores.
* **switch window**: `switch_window_s` either side of each switch, where flicker or a new identity would show.

**Which frames each score sees:**

| Score | Whole video | Each segment | Each segment core | Each switch window |
|---|---|---|---|---|
| overall_consistency | vs global prompt | | vs its local prompt | |
| clip_t | vs global prompt | | vs its local prompt | |
| prompt_match | | | vs every local prompt | |
| subject / background_consistency | yes | yes | | yes |
| temporal_flickering, motion_smoothness | yes | | | yes |
| dynamic_degree | yes | yes | | |
| aesthetic_quality, imaging_quality | yes | | | |
| i2v_subject / i2v_background | single-prompt image videos | vs image k | | |
| clip_f | yes (mean, min) | | | yes (min) |

**Plain models are split the same way.** A plain Wan 2.2 video made from the same prompts is split at the same
places, so each segment is checked against the event that should be there; a model without Prompt Relay is
not told when each event happens. MinT split its baselines the same way ("since no baselines can control
event timing, we simply set all events to have the same length"). A prompt with no or one local prompt is not
split.

## Metrics: used and not used

### VBench (16 dimensions)

| Dimension (model) | Used | Why |
|---|---|---|
| Subject consistency (DINO) | **yes** | identity drift; per segment and at the switch |
| Background consistency (CLIP) | **yes** | scene continuity; per segment and at the switch |
| Temporal flickering (frame difference) | **yes** | flicker. VBench defines it on still videos, so on moving ones it partly measures motion: read it with dynamic degree |
| Motion smoothness (AMT interpolation) | **yes** | continuity that does not punish motion |
| Dynamic degree (RAFT optical flow) | **yes** | a still video would win the consistency scores; per segment it shows each event moves |
| Aesthetic quality (LAION predictor) | **yes** | the method must not cost quality |
| Imaging quality (MUSIQ) | **yes** | over-exposure, blur, noise |
| Overall consistency (ViCLIP) | **yes** | VBench's text-video match, per segment with its local prompt |
| Object class (GRiT) | no | needs VBench's labelled prompts (VBench refuses custom prompts for it) |
| Multiple objects (GRiT) | no | same |
| Human action (UMT) | no | only Kinetics-400 action names; our events are not |
| Color (GRiT) | no | needs labels; not our question |
| Spatial relationship (GRiT) | no | needs labels; not our question |
| Scene (Tag2Text) | no | needs VBench's scene labels |
| Appearance style (CLIP) | no | the style does not change |
| Temporal style (ViCLIP) | no | camera-move wording; not our question |

### VBench++ (7 more)

| Dimension (model) | Used | Why |
|---|---|---|
| I2V subject (DINO vs input image) | **yes**, image prompts | image k vs segment k: how closely image 2 is followed |
| I2V background (DreamSim vs input image) | **yes**, image prompts | same, for the scene |
| Camera motion (CoTracker) | no | one camera move per video from 7 fixed types; our prompts have several |
| Culture fairness, gender bias, skin tone bias, safety | no | need VBench's own prompt sets; not our question |

VBench-2.0 is not used: most of its dimensions need per-prompt questions or extra 7B models.

### Extra scores (not VBench)

| Score | Switch off | From | What |
|---|---|---|---|
| `clip_t_seg{k}`, `clip_t_seg_mean`, `clip_t_global` | `--no_clip_t` | MinT, SwitchCraft | CLIP ViT-L/14 text-frame cosine, mean over segment k's core frames |
| `prompt_match_acc` | `--no_prompt_match` | ours | share of segments whose own prompt has the highest CLIP-T (event order, no extra model) |
| `clip_f_mean`, `clip_f_min`, `clip_f_switch_min` | `--no_clip_f` | SwitchCraft | CLIP cosine of neighbouring frames |
| `gen_time_s` | `--no_gen_time` | TS-Attn reports it | from the video's `.meta.json` |

## Scoring options: evaluate.py and run_all.py

```
python vbench_eval/run_all.py  --prompts FILE.json --root OUT_ROOT --out RESULTS [options]
python vbench_eval/evaluate.py --prompts FILE.json --videos_dir FOLDER --out RESULTS [options]
```

`run_all.py` scores every sub-folder of `--root` that holds `.mp4` files, as one method each, and passes the
options below to `evaluate.py`. It leaves out prompts a method has no video for (`--skip_missing`).

| Option | Default | Meaning |
|---|---|---|
| `--vbench_root` | `$VBENCH_ROOT`, else `third_party/VBench` | the VBench clone |
| `--device` | `cuda` | |
| `--margin_s` | 0.5 | seconds left out of a segment next to a switch, for prompt matching |
| `--switch_window_s` | 0.5 | seconds either side of a switch for the switch scores |
| `--min_core_frames` | 8 | the margin shrinks if a segment core would get shorter |
| `--only a,b` / `--skip a,b` | | run only / leave out these scores (names below) |
| `--no_clip_t`, `--no_prompt_match`, `--no_clip_f`, `--no_gen_time` | | extra scores off |
| `--no_i2v` | | VBench++ i2v_subject / i2v_background off |
| `--clip_model` | `ViT-L/14` | CLIP for CLIP-T / CLIP-F |
| `--vbench_local` | | VBench's 'local' mode: model files go to `~/.cache/vbench` |
| `--dry_run` | | load no models; print and save the plan (switches, clips) |
| `--force` | | score again videos already in `per_video.json` |
| `--strict` | | stop at the first error (otherwise errors are recorded per video) |
| `--skip_missing` | set by run_all | prompts with no video in the folder are left out, not errors |
| `--delete_clips` | | delete the clips after scoring (kept by default, 3-10 MB each) |
| `--method` | folder name | name stored in the results |

Score names for `--only` / `--skip`: `overall_consistency subject_consistency background_consistency
temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality i2v_subject
i2v_background clip_t prompt_match clip_f gen_time`.

## Output files and columns

`run_all.py` writes `comparison.csv` / `comparison.md` (mean per method, with the number of videos) and one
folder per method with:

| File | What |
|---|---|
| `per_video.csv` / `.json` | one row per video; the JSON also has every clip's score and the prompt-match matrix |
| `summary.csv` / `.json` | mean, standard deviation and count per column |
| `run_info.json` | settings, seconds per score, VBench commit, prompts with no video |
| `vbench_info/`, `clips/` | the files handed to VBench |

All VBench scores are 0-1 (imaging quality divided by 100, as VBench does). Higher is better except
`gen_time_s`.

| Column | Meaning |
|---|---|
| `<score>` | the whole video |
| `<score>_seg{k}`, `<score>_seg_mean` | segment k, and the mean over segments |
| `<score>_switch_min` | the lowest score among the switch windows |
| `overall_consistency_global`, `clip_t_global` | whole video vs the global prompt |
| `i2v_subject_img{k}`, `i2v_background_img{k}` | image k vs segment k |

## Cautions

* **Long prompts are cut short by the scorers.** ViCLIP (overall_consistency) reads the first 32 tokens
  (about 20-25 words), CLIP the first 77. The tool prints a note for each longer prompt.
* **Intended transitions look like failures** to CLIP-F, flickering and motion smoothness (a hand covering
  the lens, a whip pan). Every method gets the same prompts, so comparisons stay fair; tag such prompts
  (an extra field) to report them separately.
* **Generation time** includes model loading; compare methods only on the same GPUs and GPU count.
* **The same seed gives a different video on a different GPU type** (e.g. H100 vs A100).
* **Clips are re-encoded** near-losslessly (H.264, CRF 1) at the original frame rate; whole-video scores use
  the original file.
