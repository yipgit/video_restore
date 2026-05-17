#!/usr/bin/env python3
"""
Video dark-segment detector and restorer.

Stage-2 design:
- Detect abnormally dark ranges from frame luminance statistics.
- Restore with deterministic curve/gamma matching by default.
- Optionally run a Zero-DCE style Torch model when weights are provided.
- CPU/mock paths are first-class so the pipeline can be tested without CUDA.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - dependency checked at runtime
    cv2 = None

import numpy as np


@dataclass
class Segment:
    start: float
    end: float
    confidence: float
    reason: str = "brightness_drop"
    target_y: Optional[float] = None
    source_y: Optional[float] = None
    black_clip_ratio: Optional[float] = None
    shadow_entropy: Optional[float] = None
    recommended_method: str = "curve"


def require_cv2() -> None:
    if cv2 is None:
        raise SystemExit("Missing dependency: opencv-python. Install with: pip install -r requirements.txt")


def run(cmd: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("$", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: str) -> float:
    if cv2 is not None:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if fps > 0 and frames > 0:
                return float(frames / fps)
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def ffprobe_fps(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1", path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "/" in out:
        a, b = out.split("/", 1)
        return float(a) / float(b)
    return float(out)


def frame_luma_bgr(frame: np.ndarray) -> np.ndarray:
    # BT.709-ish luma on BGR input, enough for robust detection.
    b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


def entropy_u8(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def collect_stats(video: str, sample_every: float = 0.5, seek_sampling: bool = True) -> tuple[list[dict], float, int]:
    require_cv2()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or ffprobe_fps(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * sample_every)))
    stats: list[dict] = []

    def add_stat(idx: int, frame: np.ndarray) -> None:
        y = frame_luma_bgr(frame)
        dark = y[y < 40]
        stats.append({
            "t": idx / fps,
            "frame": idx,
            "mean_y": float(y.mean()),
            "median_y": float(np.median(y)),
            "p10_y": float(np.percentile(y, 10)),
            "black_clip_ratio": float((y <= 3).mean()),
            "shadow_entropy": entropy_u8(np.clip(dark, 0, 255)) if dark.size else 0.0,
        })

    if seek_sampling and total > 0:
        # Detection only needs sparse samples. Seeking avoids decoding/retrieving every
        # frame through Python/OpenCV, which is much faster for long videos when
        # sample_every is coarse. If a backend cannot seek reliably, fall back below.
        idx = 0
        while idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            add_stat(idx, frame)
            idx += step
    else:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                add_stat(idx, frame)
            idx += 1
    cap.release()
    return stats, float(fps), total


def smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if len(values) == 0 or radius <= 0:
        return values
    kernel = np.ones(radius * 2 + 1, dtype=np.float64)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, (radius, radius), mode="edge"), kernel, mode="valid")


def merge_bool_runs(times: list[float], flags: np.ndarray, sample_every: float, min_duration: float) -> list[tuple[float, float, list[int]]]:
    runs = []
    start_i = None
    for i, flag in enumerate(flags.tolist() + [False]):
        if flag and start_i is None:
            start_i = i
        elif not flag and start_i is not None:
            end_i = i - 1
            start = max(0.0, times[start_i] - sample_every * 0.5)
            end = times[end_i] + sample_every * 0.5
            if end - start >= min_duration:
                runs.append((start, end, list(range(start_i, end_i + 1))))
            start_i = None
    return runs


def detect_segments(
    video: str,
    sample_every: float = 0.5,
    min_duration: float = 0.7,
    drop_ratio: float = 0.62,
    absolute_dark: float = 55.0,
    seek_sampling: bool = True,
) -> tuple[list[Segment], dict]:
    stats, fps, total_frames = collect_stats(video, sample_every, seek_sampling)
    if not stats:
        return [], {"fps": fps, "total_frames": total_frames, "stats": []}

    y = smooth(np.array([s["median_y"] for s in stats], dtype=np.float64), radius=2)
    times = [s["t"] for s in stats]
    normal_ref = float(np.percentile(y, 75))
    baseline = max(normal_ref, 1.0)
    flags = (y < baseline * drop_ratio) & (y < absolute_dark)

    # Avoid treating an entirely dark video as one false-positive segment.
    if float(flags.mean()) > 0.80:
        flags[:] = False

    segments: list[Segment] = []
    for start, end, idxs in merge_bool_runs(times, flags, sample_every, min_duration):
        seg_y = float(np.mean(y[idxs]))
        confidence = min(0.99, max(0.1, (baseline - seg_y) / baseline))
        black_clip = float(np.mean([stats[i]["black_clip_ratio"] for i in idxs]))
        entropy = float(np.mean([stats[i]["shadow_entropy"] for i in idxs]))
        rec = "zerodce" if black_clip < 0.20 and entropy > 2.0 else "curve"
        segments.append(Segment(
            start=round(start, 3), end=round(end, 3), confidence=round(confidence, 3),
            target_y=round(baseline, 3), source_y=round(seg_y, 3),
            black_clip_ratio=round(black_clip, 4), shadow_entropy=round(entropy, 3),
            recommended_method=rec,
        ))
    meta = {"fps": fps, "total_frames": total_frames, "sample_every": sample_every, "seek_sampling": seek_sampling, "baseline_median_y": baseline, "stats": stats}
    return segments, meta


def load_segments(path: str) -> list[Segment]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("segments", [])
    return [Segment(**x) for x in data]


def in_segments(t: float, segments: list[Segment]) -> Optional[Segment]:
    for s in segments:
        if s.start <= t <= s.end:
            return s
    return None


def estimate_target_y(frame_idx: int, fps: float, segments: list[Segment], global_target: float) -> float:
    t = frame_idx / fps
    seg = in_segments(t, segments)
    if not seg:
        return global_target
    return float(seg.target_y or global_target)


def curve_enhance_bgr(frame: np.ndarray, target_y: float, max_gain: float = 3.0) -> np.ndarray:
    y = frame_luma_bgr(frame)
    src = max(float(np.median(y)), 1.0)
    gain = min(max(target_y / src, 1.0), max_gain)
    # Gamma < 1 lifts shadows. Limit it to avoid washed-out output.
    gamma = float(np.clip(math.log(max(target_y, 2.0) / 255.0) / math.log(max(src, 2.0) / 255.0), 0.45, 0.95))
    f = frame.astype(np.float32) / 255.0
    out = np.power(np.clip(f * gain * 0.85, 0, 1), gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from e


def video_encoder_args(encoder: str, crf: int, preset: str, cq: Optional[int], bitrate: Optional[str]) -> list[str]:
    """Build FFmpeg video encoder arguments.

    libx264 uses CRF. Hardware encoders generally use CQ or bitrate, so expose
    those without forcing callers to remember encoder-specific FFmpeg flags.
    """
    if encoder == "libx264":
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
    if encoder in {"h264_nvenc", "hevc_nvenc"}:
        args = ["-c:v", encoder, "-preset", preset]
        if cq is not None:
            args += ["-cq", str(cq)]
        elif bitrate:
            args += ["-b:v", bitrate]
        else:
            args += ["-cq", str(crf)]
        return args
    if encoder in {"h264_videotoolbox", "hevc_videotoolbox"}:
        args = ["-c:v", encoder]
        if bitrate:
            args += ["-b:v", bitrate]
        else:
            # VideoToolbox does not support x264-style CRF/CQ. Use a conservative
            # default when the caller only asks for hardware acceleration.
            args += ["-b:v", "8M"]
        return args
    raise SystemExit(f"Unsupported encoder: {encoder}")


class Enhancer:
    def __init__(
        self,
        method: str,
        weights: Optional[str],
        device: str,
        retinexformer_weights: Optional[str] = None,
        retinexformer_n_feat: int = 40,
        retinexformer_stage: int = 1,
        retinexformer_num_blocks: Optional[list[int]] = None,
    ):
        self.method = method
        self.weights = weights
        self.device = device
        self.model = None
        self.model_kind: Optional[str] = None
        self.retinexformer_num_blocks = retinexformer_num_blocks or [1, 2, 2]
        if method == "zerodce" or (method == "auto" and weights):
            self.model = self._load_zerodce(weights, device)
            self.model_kind = "zerodce"
        elif method == "retinexformer":
            self.model = self._load_retinexformer(
                retinexformer_weights or weights,
                device,
                n_feat=retinexformer_n_feat,
                stage=retinexformer_stage,
                num_blocks=self.retinexformer_num_blocks,
            )
            self.model_kind = "retinexformer"

    def _load_zerodce(self, weights: Optional[str], device: str):
        if not weights:
            raise SystemExit("--method zerodce requires --weights path/to/zerodce.pt. Use --method auto or --method curve for no weights.")
        try:
            import torch
        except Exception as e:
            raise SystemExit(f"PyTorch is required for --method zerodce: {e}")
        if str(weights).endswith((".pt", ".torchscript")):
            try:
                model = torch.jit.load(weights, map_location=device)
                model.eval().to(device)
                return model
            except Exception:
                # Some projects use .pt for state_dict snapshots; try that next.
                pass

        from .zerodcepp import ZeroDCEPP

        model = ZeroDCEPP(scale_factor=1)
        state = torch.load(weights, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        return model

    def _load_retinexformer(
        self,
        weights: Optional[str],
        device: str,
        n_feat: int = 40,
        stage: int = 1,
        num_blocks: Optional[list[int]] = None,
    ):
        if not weights:
            raise SystemExit("--method retinexformer requires --retinexformer-weights path/to/model.pth")
        try:
            import torch
        except Exception as e:
            raise SystemExit(f"PyTorch is required for --method retinexformer: {e}")
        try:
            from .third_party.retinexformer_arch import RetinexFormer
        except ModuleNotFoundError as e:
            if e.name == "einops":
                raise SystemExit(
                    "Retinexformer requires einops. Install it with: "
                    "pip install -r video_restore/requirements-gpu.txt or pip install einops"
                ) from e
            raise SystemExit(f"Cannot import bundled RetinexFormer architecture: {e}") from e
        except Exception as e:
            raise SystemExit(f"Cannot import bundled RetinexFormer architecture: {e}") from e

        blocks = num_blocks or [1, 2, 2]
        model = RetinexFormer(in_channels=3, out_channels=3, n_feat=n_feat, stage=stage, num_blocks=blocks)
        checkpoint = torch.load(weights, map_location=device)
        state = checkpoint
        for key in ("params", "state_dict", "model", "net", "net_g"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                k = k.replace("module.", "", 1).replace("net_g.", "", 1)
                cleaned[k] = v
            state = cleaned
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        return model

    def apply(self, frame: np.ndarray, target_y: float, recommended: str = "curve") -> np.ndarray:
        method = self.method
        if method == "auto":
            method = "zerodce" if self.model is not None and recommended == "zerodce" else "curve"
        if method == "mock":
            # Debug-only: visible deterministic transform, no ML required.
            return curve_enhance_bgr(frame, target_y, max_gain=1.4)
        if method == "curve":
            return curve_enhance_bgr(frame, target_y)
        if method == "zerodce":
            return self._apply_torch_model(frame)
        if method == "retinexformer":
            return self._apply_torch_model(frame, pad_factor=4)
        raise SystemExit(f"Unknown method: {method}")

    def _apply_torchscript(self, frame: np.ndarray) -> np.ndarray:
        return self._apply_torch_model(frame)

    def _apply_torch_model(self, frame: np.ndarray, pad_factor: int = 1) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        rgb = frame[..., ::-1].astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        h, w = x.shape[-2:]
        if pad_factor > 1:
            pad_h = (pad_factor - h % pad_factor) % pad_factor
            pad_w = (pad_factor - w % pad_factor) % pad_factor
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = self.model(x)
            else:
                y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y[:, :, :h, :w]
        out = y.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (out[..., ::-1] * 255.0 + 0.5).astype(np.uint8)


def restore_video(
    input_video: str,
    output_video: str,
    segments: list[Segment],
    method: str = "auto",
    weights: Optional[str] = None,
    device: str = "cpu",
    crf: int = 18,
    preset: str = "medium",
    encoder: str = "libx264",
    cq: Optional[int] = None,
    bitrate: Optional[str] = None,
    retinexformer_weights: Optional[str] = None,
    retinexformer_n_feat: int = 40,
    retinexformer_stage: int = 1,
    retinexformer_num_blocks: Optional[list[int]] = None,
) -> None:
    require_cv2()
    enhancer = Enhancer(
        method,
        weights,
        device,
        retinexformer_weights,
        retinexformer_n_feat,
        retinexformer_stage,
        retinexformer_num_blocks,
    )

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or ffprobe_fps(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    global_target = float(np.median([s.target_y for s in segments if s.target_y] or [96.0]))

    # Stream raw BGR frames directly into FFmpeg. This avoids relying on OpenCV's
    # platform-specific VideoWriter codecs, and lets FFmpeg copy the original audio.
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        "-i", input_video,
        "-map", "0:v:0", "-map", "1:a?",
        *video_encoder_args(encoder, crf, preset, cq, bitrate),
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", output_video,
    ]
    print("$", " ".join(cmd), file=sys.stderr)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    idx = 0
    changed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            seg = in_segments(t, segments)
            if seg:
                target = estimate_target_y(idx, fps, segments, global_target)
                # 0.35s fade at segment boundaries to prevent luminance popping.
                fade = min(1.0, max(0.0, (t - seg.start) / 0.35), max(0.0, (seg.end - t) / 0.35))
                restored = enhancer.apply(frame, target, seg.recommended_method)
                frame = cv2.addWeighted(restored, fade, frame, 1.0 - fade, 0.0)
                changed += 1
            proc.stdin.write(frame.tobytes())
            idx += 1
            if total and idx % max(1, int(fps * 5)) == 0:
                print(f"processed {idx}/{total} frames, restored {changed}", file=sys.stderr)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg encode failed with exit code {rc}")


def write_report(out: str, segments: list[Segment], meta: dict) -> None:
    payload = {"segments": [asdict(s) for s in segments], "meta": {k: v for k, v in meta.items() if k != "stats"}}
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {out}; segments={len(segments)}")


def extract_preview_clip(input_video: str, output_video: str, start: float, end: float) -> None:
    # Accurate trim for previews. Re-encode instead of stream-copying so the clip
    # starts exactly at the selected segment boundary and every comparison method
    # receives the same short source.
    cmd = [
        "ffmpeg", "-y", "-i", input_video, "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", output_video,
    ]
    run(cmd)


def write_compare_html(out_dir: Path, title: str, videos: list[tuple[str, str]], segment: Segment, clip_start: float, clip_end: float) -> Path:
    cards = []
    for label, filename in videos:
        cards.append(f"""
        <section class=\"card\">
          <h2>{escape(label)}</h2>
          <video controls preload=\"metadata\" src=\"{escape(filename)}\"></video>
        </section>
        """)
    html = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{escape(title)}</title>
<style>
  :root {{ color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  body {{ margin: 0; background: #111; color: #eee; }}
  header {{ position: sticky; top: 0; z-index: 2; background: rgba(20,20,20,.94); border-bottom: 1px solid #333; padding: 14px 18px; }}
  h1 {{ margin: 0 0 8px; font-size: 18px; }}
  .meta {{ color: #bbb; font-size: 13px; line-height: 1.5; }}
  .toolbar {{ margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }}
  button {{ background: #2f6fed; color: white; border: 0; border-radius: 8px; padding: 8px 12px; cursor: pointer; }}
  button.secondary {{ background: #333; }}
  main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; padding: 14px; }}
  .card {{ background: #1b1b1b; border: 1px solid #333; border-radius: 12px; padding: 12px; }}
  .card h2 {{ margin: 0 0 10px; font-size: 16px; }}
  video {{ width: 100%; background: #000; border-radius: 8px; }}
</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <div class=\"meta\">
    原始片段：{clip_start:.3f}s → {clip_end:.3f}s；检测段：{segment.start:.3f}s → {segment.end:.3f}s；
    confidence={segment.confidence}；source_y={segment.source_y}；target_y={segment.target_y}；推荐={escape(segment.recommended_method)}
  </div>
  <div class=\"toolbar\">
    <button onclick=\"syncPlay()\">同步播放</button>
    <button onclick=\"syncPause()\" class=\"secondary\">暂停全部</button>
    <button onclick=\"syncTime()\" class=\"secondary\">同步到第一个视频时间</button>
    <button onclick=\"setRate(.5)\" class=\"secondary\">0.5x</button>
    <button onclick=\"setRate(1)\" class=\"secondary\">1x</button>
  </div>
</header>
<main>
{''.join(cards)}
</main>
<script>
const videos = [...document.querySelectorAll('video')];
function syncTime() {{ const t = videos[0]?.currentTime || 0; videos.forEach(v => v.currentTime = t); }}
function syncPlay() {{ syncTime(); videos.forEach(v => v.play()); }}
function syncPause() {{ videos.forEach(v => v.pause()); }}
function setRate(rate) {{ videos.forEach(v => v.playbackRate = rate); }}
</script>
</body>
</html>
"""
    out = out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def compare_segment(
    input_video: str,
    segments_path: str,
    out_dir: str,
    segment_index: int = 0,
    pad: float = 1.0,
    methods: Optional[list[str]] = None,
    weights: Optional[str] = None,
    device: str = "cpu",
    crf: int = 18,
    preset: str = "veryfast",
    encoder: str = "libx264",
    cq: Optional[int] = None,
    bitrate: Optional[str] = None,
    retinexformer_weights: Optional[str] = None,
    retinexformer_n_feat: int = 40,
    retinexformer_stage: int = 1,
    retinexformer_num_blocks: Optional[list[int]] = None,
) -> Path:
    segments = load_segments(segments_path)
    if not segments:
        raise SystemExit(f"No segments found in {segments_path}")
    if segment_index < 0 or segment_index >= len(segments):
        raise SystemExit(f"--segment-index out of range: {segment_index}; total={len(segments)}")
    methods = methods or ["original", "curve", "zerodce", "retinexformer"]

    src_segment = segments[segment_index]
    duration = ffprobe_duration(input_video)
    clip_start = max(0.0, src_segment.start - pad)
    clip_end = min(duration, src_segment.end + pad)
    local_segment = Segment(**asdict(src_segment))
    local_segment.start = max(0.0, src_segment.start - clip_start)
    local_segment.end = max(local_segment.start, src_segment.end - clip_start)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    clip = out_path / "source_segment.mp4"
    extract_preview_clip(input_video, str(clip), clip_start, clip_end)
    (out_path / "segment.local.json").write_text(json.dumps({"segments": [asdict(local_segment)]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "segment.source.json").write_text(json.dumps(asdict(src_segment), ensure_ascii=False, indent=2), encoding="utf-8")

    videos: list[tuple[str, str]] = [("original", clip.name)]
    for method in methods:
        method = method.strip().lower()
        if not method or method == "original":
            continue
        if method == "zerodce" and not weights:
            print("skip zerodce: --weights is required", file=sys.stderr)
            continue
        if method == "retinexformer" and not retinexformer_weights:
            print("skip retinexformer: --retinexformer-weights is required", file=sys.stderr)
            continue
        out_video = out_path / f"{method}.mp4"
        restore_video(
            str(clip), str(out_video), [local_segment], method, weights, device,
            crf, preset, encoder, cq, bitrate,
            retinexformer_weights, retinexformer_n_feat, retinexformer_stage, retinexformer_num_blocks,
        )
        videos.append((method, out_video.name))

    html = write_compare_html(out_path, f"video_restore compare segment #{segment_index}", videos, src_segment, clip_start, clip_end)
    print(f"wrote compare UI: {html}")
    return html


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Detect and restore intentionally darkened video segments")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect")
    d.add_argument("input")
    d.add_argument("--out", default="segments.json")
    d.add_argument("--sample-every", type=float, default=0.5)
    d.add_argument("--min-duration", type=float, default=0.7)
    d.add_argument("--drop-ratio", type=float, default=0.62)
    d.add_argument("--absolute-dark", type=float, default=55.0)
    d.add_argument("--sequential-scan", action="store_true", help="Read every frame during detection instead of seeking sparse samples")

    r = sub.add_parser("restore")
    r.add_argument("input")
    r.add_argument("--segments", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--method", choices=["auto", "curve", "mock", "zerodce", "retinexformer"], default="auto")
    r.add_argument("--weights", help="Zero-DCE weights, or generic weights when the selected method accepts them")
    r.add_argument("--retinexformer-weights", help="Retinexformer .pth checkpoint, e.g. LOL_v1.pth")
    r.add_argument("--retinexformer-n-feat", type=int, default=40)
    r.add_argument("--retinexformer-stage", type=int, default=1)
    r.add_argument("--retinexformer-num-blocks", type=parse_int_list, default=[1, 2, 2], help="Comma-separated Retinexformer block counts, default: 1,2,2")
    r.add_argument("--device", default="cuda")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--preset", default="medium")
    r.add_argument("--encoder", choices=["libx264", "h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox"], default="libx264")
    r.add_argument("--cq", type=int, help="Constant-quality value for NVENC encoders; defaults to --crf")
    r.add_argument("--bitrate", help="Target video bitrate for bitrate-based hardware encoders, e.g. 8M")

    pr = sub.add_parser("process")
    pr.add_argument("input")
    pr.add_argument("--out", required=True)
    pr.add_argument("--segments-out", default="segments.json")
    pr.add_argument("--method", choices=["auto", "curve", "mock", "zerodce", "retinexformer"], default="auto")
    pr.add_argument("--weights", help="Zero-DCE weights, or generic weights when the selected method accepts them")
    pr.add_argument("--retinexformer-weights", help="Retinexformer .pth checkpoint, e.g. LOL_v1.pth")
    pr.add_argument("--retinexformer-n-feat", type=int, default=40)
    pr.add_argument("--retinexformer-stage", type=int, default=1)
    pr.add_argument("--retinexformer-num-blocks", type=parse_int_list, default=[1, 2, 2], help="Comma-separated Retinexformer block counts, default: 1,2,2")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--sample-every", type=float, default=0.5)
    pr.add_argument("--sequential-scan", action="store_true", help="Read every frame during detection instead of seeking sparse samples")
    pr.add_argument("--crf", type=int, default=18)
    pr.add_argument("--preset", default="medium")
    pr.add_argument("--encoder", choices=["libx264", "h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox"], default="libx264")
    pr.add_argument("--cq", type=int, help="Constant-quality value for NVENC encoders; defaults to --crf")
    pr.add_argument("--bitrate", help="Target video bitrate for bitrate-based hardware encoders, e.g. 8M")

    c = sub.add_parser("compare", help="Extract one detected segment, run selected methods, and write a local HTML comparison UI")
    c.add_argument("input")
    c.add_argument("--segments", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--segment-index", type=int, default=0)
    c.add_argument("--pad", type=float, default=1.0, help="Seconds of context to include before/after the segment")
    c.add_argument("--methods", default="original,curve,zerodce,retinexformer", help="Comma-separated methods: original,curve,mock,zerodce,retinexformer,auto")
    c.add_argument("--weights", help="Zero-DCE weights")
    c.add_argument("--retinexformer-weights", help="Retinexformer .pth checkpoint, e.g. LOL_v1.pth")
    c.add_argument("--retinexformer-n-feat", type=int, default=40)
    c.add_argument("--retinexformer-stage", type=int, default=1)
    c.add_argument("--retinexformer-num-blocks", type=parse_int_list, default=[1, 2, 2], help="Comma-separated Retinexformer block counts, default: 1,2,2")
    c.add_argument("--device", default="cuda")
    c.add_argument("--crf", type=int, default=18)
    c.add_argument("--preset", default="veryfast")
    c.add_argument("--encoder", choices=["libx264", "h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox"], default="libx264")
    c.add_argument("--cq", type=int, help="Constant-quality value for NVENC encoders; defaults to --crf")
    c.add_argument("--bitrate", help="Target video bitrate for bitrate-based hardware encoders, e.g. 8M")

    args = p.parse_args(argv)
    if args.cmd == "detect":
        segments, meta = detect_segments(args.input, args.sample_every, args.min_duration, args.drop_ratio, args.absolute_dark, not args.sequential_scan)
        write_report(args.out, segments, meta)
    elif args.cmd == "restore":
        device = args.device
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("CUDA not available; falling back to CPU", file=sys.stderr)
                    device = "cpu"
            except Exception:
                device = "cpu"
        restore_video(
            args.input, args.out, load_segments(args.segments), args.method, args.weights, device,
            args.crf, args.preset, args.encoder, args.cq, args.bitrate,
            args.retinexformer_weights, args.retinexformer_n_feat, args.retinexformer_stage, args.retinexformer_num_blocks,
        )
    elif args.cmd == "process":
        segments, meta = detect_segments(args.input, args.sample_every, seek_sampling=not args.sequential_scan)
        write_report(args.segments_out, segments, meta)
        device = args.device
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("CUDA not available; falling back to CPU", file=sys.stderr)
                    device = "cpu"
            except Exception:
                device = "cpu"
        restore_video(
            args.input, args.out, segments, args.method, args.weights, device,
            args.crf, args.preset, args.encoder, args.cq, args.bitrate,
            args.retinexformer_weights, args.retinexformer_n_feat, args.retinexformer_stage, args.retinexformer_num_blocks,
        )
    elif args.cmd == "compare":
        device = args.device
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("CUDA not available; falling back to CPU", file=sys.stderr)
                    device = "cpu"
            except Exception:
                device = "cpu"
        compare_segment(
            args.input, args.segments, args.out_dir, args.segment_index, args.pad,
            [m.strip() for m in args.methods.split(",")], args.weights, device,
            args.crf, args.preset, args.encoder, args.cq, args.bitrate,
            args.retinexformer_weights, args.retinexformer_n_feat, args.retinexformer_stage, args.retinexformer_num_blocks,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
