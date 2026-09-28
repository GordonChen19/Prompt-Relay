# MiniMax H3：overlapping Prompt Relay + sliding window

本目录把现有 Wan2.2 / Hunyuan1.5 的推理时控制方式接到 H3 官方推荐的
Diffusers 实现。入口是 `enable_prompt_relay(pipe)`，无需修改或重新训练权重。
新增功能默认关闭；既可分别启用，也可组合使用。

支持 `t2va`、`fl2va`、`ref2va` 三条原生工作流。CLI 支持文本与首尾帧条件；
参考图片、视频、音频使用下方 Python 接口，保留原生 reference 类型。

## 安装与运行

建议使用独立 Python 环境，在本目录运行：

```bash
pip install -r requirements.txt

# overlapping Prompt Relay + sliding window
python generate.py \
  --model_path MiniMaxAI/MiniMax-H3 \
  --prompt_filepath prompt_relay_overlap.json \
  --frame_num 124 --height 768 --width 1344 \
  --num_inference_steps 50 --seed 42 \
  --sliding_window --window_length 31 --window_stride 16 \
  --output_path outputs/h3_relay_window.mp4
```

运行需要完整 H3 权重。组件级 CPU offload 默认开启，模型参数所需的显存和
主存仍按官方要求准备；sliding window 缩小视频 attention 范围，不会减少权重大小。
推理完成后输出带原生音轨的 MP4，并保存同名 JSON 参数记录。

| 对照设置 | 参数 |
| --- | --- |
| 原生 H3 | `--prompt "完整提示词"` |
| 仅 Prompt Relay | `--prompt_filepath prompt_relay_overlap.json` |
| 仅 sliding window | `--prompt "完整提示词" --sliding_window` |
| 两者一起 | `--prompt_filepath prompt_relay_overlap.json --sliding_window` |

公平对照时，baseline 使用相同的 global/local 文本拼接结果、帧数、种子和采样步数。
可用 `build_prompt_text(config)[0]` 获取完全相同的文本。
`--sliding_window false` / `0` 显式关闭滑窗。首尾帧模式增加
`--image first.png`、`--last_image last.png` 中的一个或两个。

## 时间表

示例 JSON 的两个重叠区是 `1.8–2.6s` 和 `3.8–4.2s`。它们允许多个动作同时生效。

```json
{
  "global_prompt": "A robot and a toy car on a wooden desk, fixed camera.",
  "local_prompts": ["The robot waves.", "The car rolls from left to right."],
  "segment_intervals": [[0.0, 3.2], [2.0, 5.166666666666667]],
  "time_unit": "seconds",
  "tail_width": 0.3333333333333333,
  "epsilon": 0.001
}
```

- `global_prompt`：整段视频可见。
- `local_prompts`：按顺序与区间一一对应；相同文字也能映射到不同区间。
- `segment_intervals`：独立的左闭右开区间 `[start, end)`，可重叠、嵌套、乱序、有空隙。
  区间必须包含至少一个实际视频 latent 时间点。
- `time_unit`：`seconds` 或 `internal_frame`，默认后者。
- `tail_width`：离开区间平台后，权重衰减到 `epsilon` 的距离；单位跟随 `time_unit`。
  默认 2 个 internal frames；秒模式默认 `8/24` 秒。
- `epsilon`：默认 `1e-3`。区间外采用高斯软衰减，区间内不衰减。
- 不提供 intervals 时，`segment_lengths` 支持连续段，各段为正整数且总和必须等于
  H3 的 latent 帧数；两者都不提供时，平均分配全部 latent 帧。
- 本实现采用显式 overlapping intervals，不提供 `auto_overlap` 推断。不会把未识别的配置项静默忽略。

**H3 的时间网格与 Wan 不同。** 固定 24 FPS；本次固定版本将输出帧数向上对齐到
`17*n+5`，latent 帧数为 `5*n+2`。例如 124 输出帧约 5.167 秒，对应 37 latent 帧。
实际 latent 时间点是 `0, 1, 5, 9, 13, 17, 18, ...` 输出帧位置除以 24。
`prepare_attention_state()` 从官方 `position_ids` 中取出目标视频的时间坐标，
减掉文本/参考素材带来的起点偏移，再除以 40 转成秒，避免套用固定 4 倍压缩公式。
区间端点按这些时间点离散化。长度合法范围沿用该版本原生推理的 5–15 秒。

`window_length` / `window_stride` 同样以 **internal frames** 计数，默认 31 / 16。
这沿用了项目已有参数习惯；由于 H3 的时间网格不均匀，同样的帧数窗口对应的秒数可能略有差异。

## Python 接口

```python
import json
import torch
from diffusers import ComponentsManager, ModularPipeline
from h3_prompt_relay import enable_prompt_relay

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(
    "MiniMaxAI/MiniMax-H3", workflow="t2va", components_manager=manager
)
enable_prompt_relay(pipe)  # 在 workflow 选择之后、推理之前调用一次
pipe.load_components(dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda")

with open("prompt_relay_overlap.json", encoding="utf-8") as f:
    config = json.load(f)
result = pipe(
    prompt_relay_config=config,
    num_frames=124,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(42),
    sliding_window=True,
    window_length=31,
    window_stride=16,
    output=["videos", "audio", "sampling_rate"],
)
```

Ref2VA：将 workflow 改成 `ref2va`，在推理时传入原生
`references=[MiniMaxH3ImageReference.from_file(...), ...]`。
引用类型从 `diffusers.modular_pipelines.minimax_h3` 导入。
它会使用 `transformer_ref`，保留媒体前缀和参考条件，只给目标视频做时间控制。
传入 `prompt_relay_config` 时不要再传 `prompt`。

## 具体改动

| 文件 | 做了什么 |
| --- | --- |
| `h3_prompt_relay/schedule.py` | 拼接 global/local 文本；使用完整文本的 tokenizer 字符偏移定位每段 token；解析独立重叠区间，生成高斯衰减参数。 |
| `h3_prompt_relay/attention.py` | 从 H3 打包布局中分辨目标视频、文本与其他媒体；实现分块 attention、重叠滑窗和临时 attention processor。 |
| `h3_prompt_relay/pipeline.py` | 编码前加入提示词准备步骤；在原生去噪循环期间启用 processor，结束或异常时恢复。三种工作流共用同一实现。 |
| `generate.py` | 与已有模型相似的 JSON/滑窗命令行参数；检查时长；调用原生 H3 推理并保存音视频和参数。 |
| `tests/` | 显式 softmax 对照、CPU/CUDA 测试，以及官方 H3 小模型、布局和去噪循环测试。 |

H3 没有 Wan 那样单独的 cross-attention；文本、视频、音频共同进入 self-attention。
因此这里采用 Hunyuan 的控制方向：只在“目标视频 query → 局部文本 key”的 logit 上减去
`max(abs(t-midpoint)-half_width, 0)^2 / (2*sigma^2)`。
重叠时每个事件独立保留权重，不把多个事件归一化成互斥选择。

滑窗先保留全局 RoPE，再取视频窗口。每个窗口包含窗口内的目标视频和所有其他条件/音频 token，
重叠视频输出按覆盖次数在 FP32 中平均；文本、音频和参考条件 query 对全序列计算一次。
滑窗中的时间仍是整段视频的绝对时间。每次只创建 query chunk × key 的 bias，
不创建完整视频长度平方的 mask。

## 验证与边界

```bash
python -m unittest discover -s tests -v
# 或
python -m pytest -q tests
```

测试无需下载 H3 权重。真实模型结构使用随机初始化的小尺寸 H3 Transformer；完整去噪循环
测试仅将大型文本编码器换成确定性测试嵌入，保留官方布局、双 scheduler、Transformer 和 latent 解包。
这能验证代码与数值行为，不能代替完整权重的视频质量对比。
本地验证记录与具体环境见 [VALIDATION.md](VALIDATION.md)。

音频保留原生联合生成流程，当前不对音频 query 直接施加时间路由，也不保证音频事件时间。
音频可经联合 attention 间接受到视频变化影响。此版本只支持一个完整、未进行 sequence/context
parallel 切分的请求；启用相应并行会明确报错。支持常规组件/分组 offload，但没有针对所有
量化、缓存、编译组合做运行验证。同一 pipeline 顺序调用，避免并发共享 processor 和 scheduler。

## 固定来源

- [MiniMax H3 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3)
- [本次适配的 Diffusers 提交](https://github.com/huggingface/diffusers/tree/e0abab83b5df05de9e7abd788643c1a7c1e42e28)
- [该提交的 H3 Transformer](https://github.com/huggingface/diffusers/blob/e0abab83b5df05de9e7abd788643c1a7c1e42e28/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [该提交的 H3 布局与时间网格](https://github.com/huggingface/diffusers/blob/e0abab83b5df05de9e7abd788643c1a7c1e42e28/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py)

不依赖已失效的 `minimax-h3` 临时 Diffusers 分支。升级 Diffusers 时，应重新运行上述测试，
特别检查媒体前缀、token 顺序、时间坐标和 denoise block 接口。
