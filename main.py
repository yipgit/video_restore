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


def collect_stats(video: str, sample_every: float = 0.5) -> tuple[list[dict], float, int]:
    require_cv2()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or ffprobe_fps(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * sample_every)))
    stats: list[dict] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
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
) -> tuple[list[Segment], dict]:
    stats, fps, total_frames = collect_stats(video, sample_every)
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
    meta = {"fps": fps, "total_frames": total_frames, "sample_every": sample_every, "baseline_median_y": baseline, "stats": stats}
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


class Enhancer:
    def __init__(self, method: str, weights: Optional[str], device: str):
        self.method = method
        self.weights = weights
        self.device = device
        self.model = None
        if method == "zerodce" or (method == "auto" and weights):
            self.model = self._load_zerodce(weights, device)

    def _load_zerodce(self, weights: Optional[str], device: str):
        if not weights:
            raise SystemExit("--method zerodce requires --weights path/to/zerodce.pt. Use --method auto or --method curve for no weights.")
        try:
            import torch
        except Exception as e:
            raise SystemExit(f"PyTorch is required for --method zerodce: {e}")
        # This loader supports TorchScript or a plain state-dict with a user-supplied module later.
        # For reliability, prefer exporting Zero-DCE/Zero-DCE++ to TorchScript once, then use it here.
        model = torch.jit.load(weights, map_location=device)
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
            return self._apply_torchscript(frame)
        raise SystemExit(f"Unknown method: {method}")

    def _apply_torchscript(self, frame: np.ndarray) -> np.ndarray:
        import torch
        rgb = frame[..., ::-1].astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = self.model(x)
            else:
                y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
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
) -> None:
    require_cv2()
    enhancer = Enhancer(method, weights, device)

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
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
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

    r = sub.add_parser("restore")
    r.add_argument("input")
    r.add_argument("--segments", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--method", choices=["auto", "curve", "mock", "zerodce"], default="auto")
    r.add_argument("--weights")
    r.add_argument("--device", default="cuda")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--preset", default="medium")

    pr = sub.add_parser("process")
    pr.add_argument("input")
    pr.add_argument("--out", required=True)
    pr.add_argument("--segments-out", default="segments.json")
    pr.add_argument("--method", choices=["auto", "curve", "mock", "zerodce"], default="auto")
    pr.add_argument("--weights")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--sample-every", type=float, default=0.5)

    args = p.parse_args(argv)
    if args.cmd == "detect":
        segments, meta = detect_segments(args.input, args.sample_every, args.min_duration, args.drop_ratio, args.absolute_dark)
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
        restore_video(args.input, args.out, load_segments(args.segments), args.method, args.weights, device, args.crf, args.preset)
    elif args.cmd == "process":
        segments, meta = detect_segments(args.input, args.sample_every)
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
        restore_video(args.input, args.out, segments, args.method, args.weights, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
