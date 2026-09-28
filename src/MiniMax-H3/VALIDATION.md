# 验证记录

日期：2026-09-28。

- Diffusers：`e0abab83b5df05de9e7abd788643c1a7c1e42e28`。
- Python 3.13；PyTorch `2.11.0+cu128`；Transformers `5.17.0`。
- Hugging Face Hub `1.33.0`；Accelerate `1.15.0`。
- Windows；CUDA 数值检查使用 RTX 5060 Laptop GPU（8 GB）。
- 命令：在本目录执行 `python -m unittest discover -s tests -v`。
- 结果：**20 项测试全部通过，无跳过项**。

覆盖范围：

1. overlapping intervals、乱序/空隙、连续段、非均匀时间点、无效配置。
2. 重复文本、中英文文本、真实 byte-level BPE 的字符/token 对齐、跨内容边界拒绝。
3. 分块 attention 与逐 query 显式 softmax 对照，覆盖不同 chunk 与 window 设置。
4. 重叠事件同时保留权重；滑窗保持绝对时间；尾窗口覆盖；重叠输出平均。
5. CPU 浮点与 CUDA BF16 数值一致性、有限值检查。
6. 官方 H3 小尺寸 Transformer：全长窗口与原生 attention 一致；首帧/尾帧条件、
   图片+视频+音频引用、QKV fusion、异常恢复、text refiner 不被替换。
7. 三种原生 workflow 的安装；真实 H3 双 scheduler 去噪循环与 latent 打包/解包。
8. FL2VA 和 Ref2VA 循环中，条件视频 latent、参考音频 latent 逐值保持固定。
9. 关闭新增功能时以及启用后再次关闭时，视频/音频 latent 与原生路径逐值一致。
10. CLI 默认关闭、显式开关、无效窗口拒绝。

范围说明：小模型使用官方 H3 Transformer 类与随机权重；流水线测试将大型 Qwen
编码替换为确定性嵌入，条件素材编码替换为合成 latent。**没有加载完整 H3 权重生成视频**，
因此尚未验证完整权重的画质、事件时间遵循程度、音画同步、运行速度或峰值显存。
音视频解码和 MP4 导出沿用上游接口，此次未执行完整权重导出。
