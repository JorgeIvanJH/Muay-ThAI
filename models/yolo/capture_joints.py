import os
import sys
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
import cv2
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import models.utils as modelutils
import utils as yoloutils
import config as yolocfg


def smooth_result_keypoints(result, previous_smoothed_keypoints, alpha=yolocfg.YOLO_SMOOTHING_ALPHA):
    if result.keypoints is None:
        return None

    keypoint_data = result.keypoints.data.clone()
    current_xy = keypoint_data[..., :2]
    if previous_smoothed_keypoints is not None and previous_smoothed_keypoints.shape == current_xy.shape:
        previous_smoothed_keypoints = previous_smoothed_keypoints.to(current_xy.device)
        smoothed_xy = alpha * current_xy + (1.0 - alpha) * previous_smoothed_keypoints
    else:
        smoothed_xy = current_xy

    keypoint_data[..., :2] = smoothed_xy
    result.keypoints = result.keypoints.__class__(keypoint_data, result.orig_shape)

    return smoothed_xy.detach().clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(yolocfg.YOLO_WEIGHTS),
        help="Path to the YOLO model weights.",
    )
    parser.add_argument(
        "--source",
        default=str(yolocfg.YOLO_INPUT),
        help="Video path or webcam index.",
    )
    parser.add_argument(
        "--output",
        default=str(yolocfg.YOLO_OUTPUT),
        help="Folder where joint predictions and video are saved.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save output without opening the preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for quick test runs.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=yolocfg.YOLO_SMOOTHING_ALPHA,
        help=(
            "Temporal smoothing factor for joint xy positions. Lower values smooth more; 1.0 disables smoothing."
        ),
    )
    # TODO: Add frame-stride
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.smooth_alpha <= 1.0:
        raise ValueError("--smooth-alpha must be between 0.0 and 1.0")

    capture_source, source_label = modelutils.parse_source(args.source)
    output_paths = modelutils.build_output_paths(args.output, source_label, args.model)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    writer = None
    frame_index = 0
    previous_smoothed_keypoints = None

    with output_paths["predictions"].open("w", encoding="utf-8") as predictions_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, verbose=False)
            # temporal smoothing
            previous_smoothed_keypoints = smooth_result_keypoints(
                results[0],
                previous_smoothed_keypoints,
                args.smooth_alpha,
            )
            annotated_frame = results[0].plot()

            if writer is None:
                height, width = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_paths["video"]),
                    fourcc,
                    fps,
                    (width, height),
                )

            frame_predictions = {
                "frame_index": frame_index,
                "source": args.source,
                "model": args.model,
                "people": yoloutils.keypoints_to_people(results[0]),
                "boxes": yoloutils.boxes_to_detections(results[0]),
            }
            predictions_file.write(json.dumps(frame_predictions) + "\n")
            writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow("AI Combat Coach", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Saved predictions to {output_paths['predictions']}")
    print(f"Saved annotated video to {output_paths['video']}")


if __name__ == "__main__":
    main()
