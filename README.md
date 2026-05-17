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

检测阶段默认按 `--sample-every` 做稀疏 seek 采样，避免把每一帧都解码进 Python；如果遇到某些视频容器/编码 seek 不稳定，可退回旧的顺序扫描：

```bash
python -m video_restore.main detect input.mp4 --sequential-scan --out segments.json
```

## 硬件编码加速

默认仍使用 x264：

```bash
python -m video_restore.main process input.mp4 --out output.mp4 \
  --encoder libx264 --crf 18 --preset medium
```

NVIDIA / 3070 建议先试 NVENC：

```bash
python -m video_restore.main process input.mp4 --out output.mp4 \
  --encoder h264_nvenc --preset p4 --cq 18
```

H.265 NVENC：

```bash
python -m video_restore.main process input.mp4 --out output.mp4 \
  --encoder hevc_nvenc --preset p4 --cq 22
```

macOS VideoToolbox：

```bash
python -m video_restore.main process input.mp4 --out output.mp4 \
  --encoder h264_videotoolbox --bitrate 8M
```

## 片段级对比 UI

先检测：

```bash
python -m video_restore.main detect input.mp4 --out segments.json
```

只摘取某个检测片段，分别跑多种方法，并生成本地 HTML 对比页：

```bash
python -m video_restore.main compare input.mp4 \
  --segments segments.json \
  --segment-index 0 \
  --pad 1.0 \
  --methods original,curve,zerodce,retinexformer \
  --weights models/zerodcepp_epoch99.pth \
  --retinexformer-weights models/retinexformer_lol_v1.pth \
  --device cuda \
  --out-dir compare-seg0
```

输出目录里会包含：

- `source_segment.mp4`：从原视频摘出的短片段
- `curve.mp4` / `zerodce.mp4` / `retinexformer.mp4` 等不同方法结果
- `segment.local.json`：短片段内部使用的本地时间轴 segment
- `index.html`：并排播放对比 UI，支持同步播放/暂停

如果没有模型权重，可以先比较原片和曲线恢复：

```bash
python -m video_restore.main compare input.mp4 \
  --segments segments.json \
  --methods original,curve \
  --out-dir compare-seg0
```

`--method mock` 是轻量调试模式，用来验证检测、分段、淡入淡出、合成流程，不依赖 GPU/模型。

## GPU / Zero-DCE++ 权重

下载官方 Zero-DCE++ 权重：

```bash
mkdir -p models
curl -L -o models/zerodcepp_epoch99.pth \
  "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
```

运行：

```bash
python -m video_restore.main restore input.mp4 \
  --segments segments.json \
  --method zerodce \
  --weights models/zerodcepp_epoch99.pth \
  --device cuda \
  --out output.mp4
```

也支持 TorchScript 权重。如果使用 `--method auto --weights xxx.pth`，程序会在推荐为 `zerodce` 的片段上用模型，否则用曲线恢复。

注意：官方 Zero-DCE++ 项目许可为 Attribution-NonCommercial 4.0 International，商用前需要确认授权。

## Retinexformer 权重

项目已内置 Retinexformer 推理架构代码，运行时仍需要下载官方/兼容 checkpoint。LOL-v1 配置默认参数为：

- `--retinexformer-n-feat 40`
- `--retinexformer-stage 1`
- `--retinexformer-num-blocks 1,2,2`

单独跑 Retinexformer：

```bash
python -m video_restore.main restore input.mp4 \
  --segments segments.json \
  --method retinexformer \
  --retinexformer-weights models/retinexformer_lol_v1.pth \
  --device cuda \
  --out output-retinexformer.mp4
```

片段级对比：

```bash
python -m video_restore.main compare input.mp4 \
  --segments segments.json \
  --methods original,curve,retinexformer \
  --retinexformer-weights models/retinexformer_lol_v1.pth \
  --device cuda \
  --out-dir compare-retinexformer-seg0
```

Retinexformer 官方 repo: https://github.com/caiyuanhao1998/Retinexformer 。不同数据集权重可能需要不同的 `n_feat/stage/num_blocks`，如果加载时报 shape mismatch，按对应 YAML 的 `network_g` 调整这些参数。

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
