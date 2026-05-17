#!/usr/bin/env python3
"""Small Qt UI for segment preview/comparison.

The UI intentionally delegates processing to ``video_restore.main compare`` so
CLI and GUI stay on the same code path.
"""
from __future__ import annotations

import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QProcess, QTimer, Qt, Signal
    from PySide6.QtGui import QImage, QPixmap, QWheelEvent
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsView,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - import guard for friendly CLI error
    raise SystemExit("PySide6 is required for the Qt UI. Install with: pip install PySide6") from e


class PathPicker(QWidget):
    def __init__(self, label: str, *, file_filter: str = "All files (*)", directory: bool = False):
        super().__init__()
        self.file_filter = file_filter
        self.directory = directory
        self.edit = QLineEdit()
        self.button = QPushButton("浏览…")
        self.button.clicked.connect(self.pick)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)

    def pick(self) -> None:
        if self.directory:
            value = QFileDialog.getExistingDirectory(self, "选择目录", self.text() or os.getcwd())
        else:
            value, _ = QFileDialog.getOpenFileName(self, "选择文件", self.text() or os.getcwd(), self.file_filter)
        if value:
            self.setText(value)


class SyncImageView(QGraphicsView):
    zoomChanged = Signal(float)

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale_factor = 1.0
        self.setMinimumHeight(220)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        if self.scale_factor == 1.0:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_zoom(self, factor: float) -> None:
        factor = max(0.05, min(20.0, factor))
        self.resetTransform()
        self.scale(factor, factor)
        self.scale_factor = factor

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.pixmap_item.pixmap().isNull():
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.scale_factor * (1.25 if delta > 0 else 0.8)
        self.set_zoom(factor)
        self.zoomChanged.emit(self.scale_factor)
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("video_restore Segment Compare")
        self.resize(1180, 760)
        self.process: QProcess | None = None
        self.segments: list[dict[str, Any]] = []
        self.enhancer_cache: dict[tuple[Any, ...], Any] = {}
        self.syncing_views = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QGridLayout(root)

        self.video_picker = PathPicker("视频", file_filter="Video files (*.mp4 *.mov *.mkv *.avi);;All files (*)")
        self.segments_picker = PathPicker("Segments", file_filter="JSON files (*.json);;All files (*)")
        self.out_picker = PathPicker("输出目录", directory=True)
        default_out = str(Path.cwd() / "compare-ui")
        self.out_picker.setText(default_out)

        self.load_button = QPushButton("加载 segments")
        self.load_button.clicked.connect(self.load_segments)

        top = QVBoxLayout()
        top.addWidget(self.video_picker)
        top.addWidget(self.segments_picker)
        top.addWidget(self.out_picker)
        top.addWidget(self.load_button)
        top_box = QGroupBox("项目")
        top_box.setLayout(top)
        layout.addWidget(top_box, 0, 0, 1, 2)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["#", "start", "end", "dur", "conf", "source_y", "target_y", "method"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.update_selected_label)
        layout.addWidget(self.table, 1, 0, 1, 1)

        frame_box = QGroupBox("单帧预览 / 图片对比")
        frame_layout = QVBoxLayout(frame_box)
        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        self.frame_label = QLabel("frame: -")
        self.preview_button = QPushButton("刷新当前帧")
        self.preview_button.clicked.connect(self.update_frame_preview)
        self.fit_button = QPushButton("重置缩放")
        self.fit_button.clicked.connect(self.fit_preview_views)
        slider_row.addWidget(self.frame_label)
        slider_row.addWidget(self.frame_slider, 1)
        slider_row.addWidget(self.preview_button)
        slider_row.addWidget(self.fit_button)
        frame_layout.addLayout(slider_row)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_frame_preview)
        self.image_grid = QGridLayout()
        self.image_views: dict[str, SyncImageView] = {}
        for i, name in enumerate(["original", "curve", "mock", "zerodce", "retinexformer"]):
            box = QGroupBox(name)
            box_layout = QVBoxLayout(box)
            view = SyncImageView(name)
            view.zoomChanged.connect(self.sync_zoom)
            view.horizontalScrollBar().valueChanged.connect(lambda value, src=view: self.sync_scroll(src, horizontal=True))
            view.verticalScrollBar().valueChanged.connect(lambda value, src=view: self.sync_scroll(src, horizontal=False))
            self.image_views[name] = view
            box_layout.addWidget(view)
            self.image_grid.addWidget(box, i // 2, i % 2)
        frame_layout.addLayout(self.image_grid)
        layout.addWidget(frame_box, 2, 0, 2, 1)

        options_box = QGroupBox("对比设置")
        form = QFormLayout(options_box)
        self.selected_label = QLabel("未选择")
        form.addRow("当前片段", self.selected_label)

        self.pad_spin = QDoubleSpinBox()
        self.pad_spin.setRange(0, 30)
        self.pad_spin.setSingleStep(0.5)
        self.pad_spin.setValue(1.0)
        form.addRow("前后 padding 秒", self.pad_spin)

        self.device_edit = QLineEdit("cuda")
        self.encoder_edit = QLineEdit("libx264")
        self.preset_edit = QLineEdit("veryfast")
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(18)
        form.addRow("device", self.device_edit)
        form.addRow("encoder", self.encoder_edit)
        form.addRow("preset", self.preset_edit)
        form.addRow("crf", self.crf_spin)

        methods = QWidget()
        methods_layout = QVBoxLayout(methods)
        methods_layout.setContentsMargins(0, 0, 0, 0)
        self.method_checks: dict[str, QCheckBox] = {}
        for name, checked in [("original", True), ("curve", True), ("mock", False), ("zerodce", False), ("retinexformer", True)]:
            cb = QCheckBox(name)
            cb.setChecked(checked)
            self.method_checks[name] = cb
            methods_layout.addWidget(cb)
        form.addRow("methods", methods)

        self.zerodce_picker = PathPicker("ZeroDCE", file_filter="Weights (*.pth *.pt *.torchscript);;All files (*)")
        existing_zd = Path.cwd() / "models" / "zerodcepp_epoch99.pth"
        if existing_zd.exists():
            self.zerodce_picker.setText(str(existing_zd))
        form.addRow(self.zerodce_picker)

        self.retinex_picker = PathPicker("Retinex", file_filter="Weights (*.pth *.pt);;All files (*)")
        form.addRow(self.retinex_picker)

        self.n_feat_spin = QSpinBox()
        self.n_feat_spin.setRange(1, 512)
        self.n_feat_spin.setValue(40)
        self.stage_spin = QSpinBox()
        self.stage_spin.setRange(1, 8)
        self.stage_spin.setValue(1)
        self.blocks_edit = QLineEdit("1,2,2")
        form.addRow("Retinex n_feat", self.n_feat_spin)
        form.addRow("Retinex stage", self.stage_spin)
        form.addRow("Retinex blocks", self.blocks_edit)

        self.run_button = QPushButton("生成对比")
        self.run_button.clicked.connect(self.run_compare)
        self.open_button = QPushButton("打开输出目录")
        self.open_button.clicked.connect(self.open_out_dir)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.open_button)
        form.addRow(buttons)
        layout.addWidget(options_box, 1, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.log, 2, 1, 2, 1)

        layout.setColumnStretch(0, 4)
        layout.setColumnStretch(1, 2)
        layout.setRowStretch(3, 1)

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip())

    def load_segments(self) -> None:
        path = self.segments_picker.text()
        if not path:
            self.error("请选择 segments.json")
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            segments = data.get("segments", data) if isinstance(data, dict) else data
            if not isinstance(segments, list):
                raise ValueError("segments JSON must contain a list or {'segments': [...]}" )
        except Exception as e:
            self.error(f"读取 segments 失败：{e}")
            return

        self.segments = segments
        self.table.setRowCount(len(segments))
        for i, seg in enumerate(segments):
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            values = [
                i,
                f"{start:.3f}",
                f"{end:.3f}",
                f"{max(0, end - start):.3f}",
                seg.get("confidence", ""),
                seg.get("source_y", ""),
                seg.get("target_y", ""),
                seg.get("recommended_method", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, i)
                self.table.setItem(i, col, item)
        self.table.resizeColumnsToContents()
        if self.segments:
            self.table.selectRow(0)
        self.append_log(f"loaded {len(self.segments)} segments from {path}")

    def selected_index(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        return rows[0].row()

    def update_selected_label(self) -> None:
        idx = self.selected_index()
        if idx is None or idx >= len(self.segments):
            self.selected_label.setText("未选择")
            self.frame_slider.setEnabled(False)
            return
        seg = self.segments[idx]
        self.selected_label.setText(f"#{idx}: {seg.get('start')} → {seg.get('end')}")
        self.configure_frame_slider(seg)

    def selected_methods(self) -> list[str]:
        return [name for name, cb in self.method_checks.items() if cb.isChecked()]

    def configure_frame_slider(self, seg: dict[str, Any]) -> None:
        video = self.video_picker.text()
        if not video or not Path(video).exists():
            self.frame_slider.setEnabled(False)
            self.frame_label.setText("frame: 请选择视频")
            return
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            start_frame = max(0, int(float(seg.get("start", 0)) * fps))
            end_frame = max(start_frame, int(float(seg.get("end", 0)) * fps))
            if total:
                end_frame = min(end_frame, max(0, total - 1))
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(start_frame, end_frame)
            self.frame_slider.setValue((start_frame + end_frame) // 2)
            self.frame_slider.blockSignals(False)
            self.frame_slider.setEnabled(True)
            self.update_frame_label()
        except Exception as e:
            self.frame_slider.setEnabled(False)
            self.frame_label.setText(f"frame: 初始化失败 {e}")

    def frame_slider_changed(self) -> None:
        self.update_frame_label()
        # Debounce while dragging: only run models shortly after movement stops.
        self.preview_timer.start(250)

    def update_frame_label(self) -> None:
        video = self.video_picker.text()
        fps = 0.0
        if video and Path(video).exists():
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(video)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                cap.release()
            except Exception:
                fps = 0.0
        frame = self.frame_slider.value()
        t = frame / fps if fps else 0.0
        self.frame_label.setText(f"frame: {frame} / {t:.3f}s")

    def selected_segment_obj(self):
        idx = self.selected_index()
        if idx is None or idx >= len(self.segments):
            return None
        from video_restore.main import Segment
        seg = self.segments[idx]
        return Segment(**seg)

    def read_current_frame(self):
        video = self.video_picker.text()
        if not video or not Path(video).exists():
            raise RuntimeError("请选择有效视频文件")
        import cv2  # type: ignore
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_slider.value())
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read frame: {self.frame_slider.value()}")
        return frame

    def frame_to_pixmap(self, frame) -> QPixmap:
        import numpy as np
        rgb = np.ascontiguousarray(frame[..., ::-1])
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def get_enhancer(self, method: str):
        from video_restore.main import Enhancer, parse_int_list
        key = (
            method,
            self.zerodce_picker.text(),
            self.retinex_picker.text(),
            self.device_edit.text().strip() or "cuda",
            self.n_feat_spin.value(),
            self.stage_spin.value(),
            self.blocks_edit.text().strip() or "1,2,2",
        )
        if key not in self.enhancer_cache:
            self.enhancer_cache[key] = Enhancer(
                method,
                self.zerodce_picker.text() or None,
                self.device_edit.text().strip() or "cuda",
                self.retinex_picker.text() or None,
                self.n_feat_spin.value(),
                self.stage_spin.value(),
                parse_int_list(self.blocks_edit.text().strip() or "1,2,2"),
            )
        return self.enhancer_cache[key]

    def update_frame_preview(self) -> None:
        if not self.frame_slider.isEnabled():
            return
        methods = self.selected_methods()
        if not methods:
            self.error("至少选择一个 method")
            return
        seg = self.selected_segment_obj()
        if seg is None:
            self.error("请选择一个 segment")
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            frame = self.read_current_frame()
            target_y = float(seg.target_y or 96.0)
            for name, view in self.image_views.items():
                if name not in methods:
                    view.set_pixmap(QPixmap())
                    continue
                if name == "original":
                    out = frame
                else:
                    if name == "zerodce" and not self.zerodce_picker.text():
                        self.append_log("skip zerodce frame preview: missing weights")
                        view.set_pixmap(QPixmap())
                        continue
                    if name == "retinexformer" and not self.retinex_picker.text():
                        self.append_log("skip retinexformer frame preview: missing weights")
                        view.set_pixmap(QPixmap())
                        continue
                    enhancer = self.get_enhancer(name)
                    out = enhancer.apply(frame, target_y, seg.recommended_method)
                view.set_pixmap(self.frame_to_pixmap(out))
            self.append_log(f"updated frame preview: frame={self.frame_slider.value()}, methods={','.join(methods)}")
        except Exception as e:
            self.error(f"单帧预览失败：{e}")
            self.append_log(f"frame preview failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def fit_preview_views(self) -> None:
        for view in self.image_views.values():
            view.scale_factor = 1.0
            if not view.scene.sceneRect().isEmpty():
                view.fitInView(view.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def sync_zoom(self, factor: float) -> None:
        if self.syncing_views:
            return
        self.syncing_views = True
        try:
            for view in self.image_views.values():
                sender = self.sender()
                if view is not sender:
                    view.set_zoom(factor)
        finally:
            self.syncing_views = False

    def sync_scroll(self, source: SyncImageView, *, horizontal: bool) -> None:
        if self.syncing_views:
            return
        self.syncing_views = True
        try:
            src_bar = source.horizontalScrollBar() if horizontal else source.verticalScrollBar()
            denom = max(1, src_bar.maximum() - src_bar.minimum())
            ratio = (src_bar.value() - src_bar.minimum()) / denom
            for view in self.image_views.values():
                if view is source:
                    continue
                bar = view.horizontalScrollBar() if horizontal else view.verticalScrollBar()
                bar.setValue(int(bar.minimum() + ratio * max(1, bar.maximum() - bar.minimum())))
        finally:
            self.syncing_views = False

    def run_compare(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.error("已有任务正在运行")
            return
        video = self.video_picker.text()
        segments = self.segments_picker.text()
        out_root = Path(self.out_picker.text() or "compare-ui")
        idx = self.selected_index()
        if not video or not Path(video).exists():
            self.error("请选择有效视频文件")
            return
        if not segments or not Path(segments).exists():
            self.error("请选择有效 segments.json")
            return
        if idx is None:
            self.error("请选择一个 segment")
            return
        methods = self.selected_methods()
        if not methods:
            self.error("至少选择一个 method")
            return

        out_dir = out_root / f"seg{idx:03d}"
        args = [
            "-m", "video_restore.main", "compare", video,
            "--segments", segments,
            "--segment-index", str(idx),
            "--pad", str(self.pad_spin.value()),
            "--methods", ",".join(methods),
            "--out-dir", str(out_dir),
            "--device", self.device_edit.text().strip() or "cuda",
            "--encoder", self.encoder_edit.text().strip() or "libx264",
            "--preset", self.preset_edit.text().strip() or "veryfast",
            "--crf", str(self.crf_spin.value()),
            "--retinexformer-n-feat", str(self.n_feat_spin.value()),
            "--retinexformer-stage", str(self.stage_spin.value()),
            "--retinexformer-num-blocks", self.blocks_edit.text().strip() or "1,2,2",
        ]
        if self.zerodce_picker.text():
            args += ["--weights", self.zerodce_picker.text()]
        if self.retinex_picker.text():
            args += ["--retinexformer-weights", self.retinex_picker.text()]

        self.run_button.setEnabled(False)
        self.log.clear()
        self.append_log("$ " + sys.executable + " " + " ".join(args))
        self.process = QProcess(self)
        # Run from the parent directory so `python -m video_restore.main` can
        # import the package while preserving the user's normal environment
        # (PATH, CUDA variables, etc.) for ffmpeg and torch.
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self.process.readyReadStandardOutput.connect(lambda: self.append_log(bytes(self.process.readAllStandardOutput()).decode(errors="replace")))
        self.process.readyReadStandardError.connect(lambda: self.append_log(bytes(self.process.readAllStandardError()).decode(errors="replace")))
        self.process.finished.connect(lambda code, status: self.compare_finished(code, status, out_dir))
        self.process.start(sys.executable, args)

    def compare_finished(self, code: int, _status: QProcess.ExitStatus, out_dir: Path) -> None:
        self.run_button.setEnabled(True)
        self.append_log(f"\nfinished with code {code}")
        html = out_dir / "index.html"
        if code == 0 and html.exists():
            webbrowser.open(html.resolve().as_uri())
        elif code != 0:
            self.error("生成对比失败，请看日志")

    def open_out_dir(self) -> None:
        out = Path(self.out_picker.text() or "compare-ui")
        out.mkdir(parents=True, exist_ok=True)
        webbrowser.open(out.resolve().as_uri())

    def error(self, message: str) -> None:
        QMessageBox.warning(self, "video_restore", message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
