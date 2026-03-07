[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://gordonchen19.github.io/STENCIL.github.io/static/pdfs/chen.pdf)
[![Project Page](https://img.shields.io/badge/Project-Website-green?logo=googlechrome&logoColor=green)](https://gordonchen19.github.io/STENCIL.github.io/)

<h1 align="center">Prompt Relay: Fine-Grained Temporal Prompt Routing for Multi-Event Video Generation</h1>

<p align="center">
  <a href="https://gordonchen19.github.io">Gordon Chen</a>
</p>


---

## :mega: Overview

Video diffusion models have achieved remarkable progress in generating high-quality videos. However, these models struggle to represent the temporal succession of multiple events in real-world videos and lack explicit mechanisms to control when semantic concepts appear, how long they persist, and the order in which multiple events occur. Such control is especially important for movie-grade synthesis, where coherent storytelling depends on precise timing, duration, and transitions between events. When using a single paragraph-style prompt to describe a sequence of complex events, models often exhibit temporal entanglement, where semantics intended for different moments interfere with one another, resulting in poor text-video alignment. 

**Prompt Relay** is an **inference-time, training-free, plug-and-play** method for fine-grained temporal control in video generation. Given a sequence of temporally constrained prompts, Prompt Relay routes each textual instruction to its intended temporal segment by modifying the cross-attention mechanism with a distance-based penalty. This enforces temporal locality, suppresses interference across segments, and enables multiple events to unfold in the correct order.

Our method preserves overall video coherence while ensuring that each prompt primarily influences only its assigned time interval.

> **Note:** This repository is under active development.

---

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

Here, $C(Q, K)$ is a distance-based penalty that suppresses attention between latent queries and prompt tokens that fall outside the intended temporal segment. This encourages each prompt to guide only its designated region of the video, while preventing semantic leakage into neighboring intervals.

This makes Prompt Relay a simple yet effective way to retrofit temporal control onto existing video generation pipelines without retraining the underlying model.

---

## Qualitative Results

Prompt Relay improves:
- **temporal alignment**, by keeping each instruction localized to its assigned segment,
- **transition naturalness**, by ensuring smooth event handoffs across time,
- **visual quality**, by reducing unnecessary competition in cross-attention.

On the examples shown in the project page, Prompt Relay consistently outperforms baseline prompting strategies and remains competitive with recent strong models such as **Kling 3.0**. In particular, **Wan 2.2 + Prompt Relay** often produces stronger visual structure and more stable multi-event generation than the base Wan 2.2 model.
