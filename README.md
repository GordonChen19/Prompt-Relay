[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://gordonchen19.github.io/STENCIL.github.io/static/pdfs/chen.pdf)
[![Project Page](https://img.shields.io/badge/Project-Website-green?logo=googlechrome&logoColor=green)](https://gordonchen19.github.io/Prompt-Relay/)

> **Note:** This repository is under construction.

<h1 align="center">
  <img src="static/images/application/Logo.png" alt="Prompt Relay logo" width="56" />
  Prompt Relay:  Inference-Time Temporal Prompt Routing For Multi-Event Video Generation
</h1>

<p align="center">
  <a href="https://gordonchen19.github.io">Gordon Chen</a>,
   <a href="https://ziqihuangg.github.io">Ziqi Huang</a>,
   <a href="https://liuziwei7.github.io/team.html">Ziwei Liu</a>
</p>


## :mega: Overview

Video diffusion models have achieved remarkable progress in generating high-quality videos. However, these models struggle to represent the temporal succession of multiple events in real-world videos and lack explicit mechanisms to control when semantic concepts appear, how long they persist, and the order in which multiple events occur. Such control is especially important for movie-grade synthesis, where coherent storytelling depends on precise timing, duration, and transitions between events. When using a single paragraph-style prompt to describe a sequence of complex events, models often exhibit temporal entanglement, where semantics intended for different moments interfere with one another, resulting in poor text-video alignment. 

**Prompt Relay** is an **inference-time, training-free, plug-and-play** method for fine-grained temporal control in video generation. Given a sequence of temporally constrained prompts, Prompt Relay routes each textual instruction to its intended temporal segment by modifying the cross-attention mechanism with a distance-based penalty.


## Pipeline

The overall goal is to generate a video from a sequence of temporally constrained prompts:

$$
\{(p_s, t_s^{start}, t_s^{end})\}_{s=1}^{N}
$$

where each prompt $p_s$ should be realized only within its designated temporal interval $[t_s^{start}, t_s^{end}]$.

Prompt Relay achieves this by introducing a temporal routing prior directly into cross-attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} - C(Q, K)\right)V
$$

Here, $C(Q, K)$ is a distance-based penalty that suppresses attention between latent queries inside the segment and prompt tokens that fall outside the intended temporal segment. This encourages each prompt to guide only its designated region of the video, while preventing semantic leakage into neighboring intervals.

This makes Prompt Relay a simple yet effective way to retrofit temporal control onto existing video generation pipelines without retraining the underlying model.

## Qualitative Results

Prompt Relay improves:
- **temporal alignment**, by keeping each instruction localized to its assigned segment,
- **transition naturalness**, by ensuring smooth event handoffs across time,
- **visual quality**, by reducing unnecessary competition in cross-attention.

Prompt Relay consistently outperforms baseline prompting strategies and remains competitive with recent strong models such as **Kling 3.0**. In particular, **Wan 2.2 + Prompt Relay** often produces stronger visual structure and more stable multi-event generation than the base Wan 2.2 model.

## Qualitative Comparison

The table below compares the two variants for each video shown on the [project page](https://gordonchen19.github.io/Prompt-Relay/).

<table>
  <thead>
    <tr>
      <th>Wan2.2</th>
      <th>Wan2.2 + Prompt Relay (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><img src="static/videos/eagle/wan22.gif" alt="Eagle Wan2.2" width="100%"></td>
      <td width="50%"><img src="static/videos/eagle/scene_transition_1.gif" alt="Eagle Prompt Relay" width="100%"></td>
    </tr>
    <tr>
      <td width="50%"><img src="static/videos/caveman/wan22.gif" alt="Caveman Wan2.2" width="100%"></td>
      <td width="50%"><img src="static/videos/caveman/scene_transition_2.gif" alt="Caveman Prompt Relay" width="100%"></td>
    </tr>
    <tr>
      <td width="50%"><img src="static/videos/hkcanyon/wan22_new.gif" alt="HK Canyon Wan2.2" width="100%"></td>
      <td width="50%"><img src="static/videos/hkcanyon/pr.gif" alt="HK Canyon Prompt Relay" width="100%"></td>
    </tr>
    <tr>
      <td width="50%"><img src="static/videos/child/wan22.gif" alt="Child Wan2.2" width="100%"></td>
      <td width="50%"><img src="static/videos/child/prompt_relay.gif" alt="Child Prompt Relay" width="100%"></td>
    </tr>
  </tbody>
</table>


## Usage

Prompt Relay takes as input a **global prompt**, a list of **local prompts**, and their corresponding **segment lengths**.

Users can define their prompts in:

```bash
Wan2.2/prompts.json
```

and then run:

```bash
python dbl/Wan2.2/generate.py \
  --task t2v-A14B \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --frame_num 81 \
  --size "832*480" \
  --prompt_filepath dbl/Wan2.2/prompts.json\
```

The rest works the same as Wan's original repository.
