"""Two-window presentation for real-time action inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from models.action_detection.config import CLASS_COLORS
from models.action_detection.realtime.actions import ActionPrediction
from models.action_detection.realtime.telemetry import PerformanceSnapshot


VIDEO_WINDOW_TITLE = "Muay-ThAI - video"
METRICS_WINDOW_TITLE = "Muay-ThAI - metrics"
METRICS_PANEL_WIDTH = 620


@dataclass(frozen=True)
class _MetricsRow:
    """One line of text in the metrics window."""

    text: str
    color: tuple[int, int, int]
    font_scale: float
    thickness: int


def _metrics_rows(
    predictions: Sequence[ActionPrediction],
    analytics_lines: Sequence[str],
    performance: PerformanceSnapshot,
) -> tuple[_MetricsRow, ...]:
    """Format model, analytics and performance values as display rows.

    Usage: Inference only.
    """

    rows = []
    for prediction in predictions:
        color_rgb = CLASS_COLORS.get(
            prediction.class_name,
            (255, 200, 0),
        )
        rows.append(
            _MetricsRow(
                text=(
                    f"{prediction.classification_task.capitalize()}: "
                    f"{prediction.class_name}  "
                    f"{prediction.probability:.1%}"
                ),
                color=tuple(
                    int(channel) for channel in reversed(color_rgb)
                ),
                font_scale=0.78,
                thickness=2,
            )
        )
    rows.extend(
        _MetricsRow(line, (235, 235, 235), 0.56, 1)
        for line in analytics_lines
    )
    rows.extend(
        (
            _MetricsRow(
                text=(
                    f"capture {performance.capture_fps:.1f} | "
                    f"pose {performance.pose_fps:.1f} | actions "
                    f"{performance.action_fps:.1f} FPS"
                ),
                color=(220, 220, 220),
                font_scale=0.52,
                thickness=1,
            ),
            _MetricsRow(
                text=(
                    f"latency {performance.end_to_end_p95_ms:.0f} ms p95 | "
                    f"dropped {performance.dropped_frames}"
                ),
                color=(220, 220, 220),
                font_scale=0.52,
                thickness=1,
            ),
        )
    )
    return tuple(rows)


def _fit_row(row: _MetricsRow, max_width: int) -> _MetricsRow:
    """Shrink an unusually long row enough to fit the metrics window.

    Usage: Inference only.
    """

    text_width = cv2.getTextSize(
        row.text,
        cv2.FONT_HERSHEY_SIMPLEX,
        row.font_scale,
        row.thickness,
    )[0][0]
    if text_width <= max_width:
        return row
    return _MetricsRow(
        text=row.text,
        color=row.color,
        font_scale=max(0.36, row.font_scale * max_width / text_width),
        thickness=row.thickness,
    )


def render_metrics_panel(
    predictions: Sequence[ActionPrediction],
    analytics_lines: Sequence[str],
    performance: PerformanceSnapshot,
) -> np.ndarray:
    """Render a readable black image for the separate metrics window.

    Usage: Inference only.
    """

    padding = 16
    row_gap = 8
    rows = tuple(
        _fit_row(row, METRICS_PANEL_WIDTH - 2 * padding)
        for row in _metrics_rows(
            predictions,
            analytics_lines,
            performance,
        )
    )
    layout = []
    panel_height = 2 * padding
    for row in rows:
        (_, text_height), baseline = cv2.getTextSize(
            row.text,
            cv2.FONT_HERSHEY_SIMPLEX,
            row.font_scale,
            row.thickness,
        )
        layout.append((row, text_height, baseline))
        panel_height += text_height + baseline + row_gap

    panel = np.full(
        (panel_height, METRICS_PANEL_WIDTH, 3),
        20,
        dtype=np.uint8,
    )
    text_y = padding
    for row, text_height, baseline in layout:
        text_y += text_height
        cv2.putText(
            panel,
            row.text,
            (padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            row.font_scale,
            row.color,
            row.thickness,
            cv2.LINE_AA,
        )
        text_y += baseline + row_gap
    return panel


def overlay_metrics_panel(
    frame: np.ndarray,
    panel: np.ndarray,
) -> np.ndarray:
    """Add metrics to a copy used only for the saved annotated video.

    Usage: Inference only.
    """

    frame_height, frame_width = frame.shape[:2]
    panel_height, panel_width = panel.shape[:2]
    scale = min(
        1.0,
        frame_width * 0.65 / panel_width,
        frame_height * 0.42 / panel_height,
    )
    target_size = (
        max(1, round(panel_width * scale)),
        max(1, round(panel_height * scale)),
    )
    resized_panel = cv2.resize(
        panel,
        target_size,
        interpolation=cv2.INTER_AREA,
    )
    margin = max(4, round(min(frame_width, frame_height) * 0.01))
    target_width, target_height = target_size
    output = frame.copy()
    output[
        margin : margin + target_height,
        margin : margin + target_width,
    ] = resized_panel
    return output


def fit_inside(
    width: int,
    height: int,
    *,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    """Fit dimensions inside bounds without changing their proportions.

    Usage: Inference only.
    """

    scale = min(1.0, max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


class InferenceWindows:
    """Own the independent video and metrics OpenCV windows."""

    def __init__(self) -> None:
        """Create lazy window state without touching the GUI backend.

        Usage: Inference only.
        """

        self._opened = False

    def _open(
        self,
        video_frame: np.ndarray,
        metrics_panel: np.ndarray,
    ) -> None:
        """Create and place both windows on their first rendered frame.

        Usage: Inference only.
        """

        cv2.namedWindow(
            VIDEO_WINDOW_TITLE,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        cv2.namedWindow(
            METRICS_WINDOW_TITLE,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        frame_height, frame_width = video_frame.shape[:2]
        metrics_height, metrics_width = metrics_panel.shape[:2]
        window_width, window_height = fit_inside(
            frame_width,
            frame_height,
            max_width=720,
            max_height=720,
        )
        cv2.resizeWindow(VIDEO_WINDOW_TITLE, window_width, window_height)
        cv2.resizeWindow(
            METRICS_WINDOW_TITLE,
            metrics_width,
            metrics_height,
        )
        cv2.moveWindow(VIDEO_WINDOW_TITLE, 0, 0)
        cv2.moveWindow(METRICS_WINDOW_TITLE, window_width + 12, 0)
        self._opened = True

    def show(self, video_frame: np.ndarray, metrics_panel: np.ndarray) -> bool:
        """Update both windows and report whether the user pressed q.

        Usage: Inference only.
        """

        if not self._opened:
            self._open(video_frame, metrics_panel)
        cv2.imshow(VIDEO_WINDOW_TITLE, video_frame)
        cv2.imshow(METRICS_WINDOW_TITLE, metrics_panel)
        return cv2.waitKey(1) & 0xFF == ord("q")


def close_inference_windows() -> None:
    """Close any video and metrics windows created during inference.

    Usage: Inference only.
    """

    cv2.destroyAllWindows()
