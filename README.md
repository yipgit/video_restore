# Video Restore

检测视频中异常降低亮度的片段，并只对这些片段做亮度恢复，再把原音频合回去。

## 安装

```bash
python3 -m venv video_restore/.venv
source video_restore/.venv/bin/activate
pip install -r video_restore/requirements.txt
# GPU/Zero-DCE 可选依赖，建议在 3070 机器上用 Python 3.10-3.12 安装：
# pip install -r video_restore/requirements-gpu.txt
```

系统需要 `ffmpeg` / `ffprobe`。

## CPU 调试流程

```bash
python -m video_restore.main detect input.mp4 --out segments.json
python -m video_restore.main restore input.mp4 --segments segments.json --method curve --device cpu --out output.mp4
```

也可以一条命令：

```bash
python -m video_restore.main process input.mp4 --method curve --device cpu --out output.mp4
```

`--method mock` 是轻量调试模式，用来验证检测、分段、淡入淡出、合成流程，不依赖 GPU/模型。

## GPU / Zero-DCE 路径

建议把 Zero-DCE/Zero-DCE++ 导出成 TorchScript：

```bash
python -m video_restore.main restore input.mp4 \
  --segments segments.json \
  --method zerodce \
  --weights zerodce_torchscript.pt \
  --device cuda \
  --out output.mp4
```

如果使用 `--method auto --weights xxx.pt`，程序会在推荐为 `zerodce` 的片段上用模型，否则用曲线恢复。

## 输出 segments.json

每段包含：

- `start` / `end`: 秒
- `confidence`: 暗化检测置信度
- `target_y` / `source_y`: 目标/当前亮度估计
- `black_clip_ratio`: 黑位截断比例
- `shadow_entropy`: 暗部信息量估计
- `recommended_method`: `curve` 或 `zerodce`

## 没有 GPU 怎么调试？

1. 用 CPU 跑 `detect` 验证片段定位。
2. 用 `curve` 或 `mock` 跑完整 restore，验证帧读写、时间轴、音频合成、边界淡入淡出。
3. 用很小的测试视频做回归测试。
4. GPU 只验证两件事：TorchScript 模型能加载、显存/速度/视觉效果是否达标。

这样可以把 80% 的工程问题在无 GPU 环境解决，避免把调试全部绑在 3070 上。
