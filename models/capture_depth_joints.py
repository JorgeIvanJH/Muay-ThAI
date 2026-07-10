import argparse
import json
import sys
from multiprocessing import Event, Process, Queue, freeze_support
from pathlib import Path
from queue import Empty, Full

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import models.utils as modelutils
import models.config as modelcfg
from models.mde import config as mdecfg
from models.mde import utils as mdeutils
from models.yolo import config as yolocfg
from models.yolo import utils as yoloutils


FRAME_QUEUE_SIZE = 2
QUEUE_PUT_TIMEOUT_SECONDS = 0.2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run metric MDE and YOLO pose tracking in parallel processes."
    )
    parser.add_argument(
        "--source",
        default=str(modelcfg.YOLOMDE_INPUT),
        help="Video path or webcam index.",
    )
    parser.add_argument(
        "--depth-model",
        default=str(mdecfg.MDE_HF_WEIGHTS),
        help="Path to the Depth Anything-V2-metric model weights.",
    )
    parser.add_argument(
        "--yolo-model",
        default=str(yolocfg.YOLO_WEIGHTS),
        help="Path to the YOLO pose model weights.",
    )
    parser.add_argument(
        "--output",
        default=str(modelcfg.YOLOMDE_OUTPUT),
        help="Folder where predictions and videos are saved.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save output without opening preview windows.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for quick test runs.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=mdecfg.MDE_MAX_DEPTH,
        help="Maximum metric depth used by the depth model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=modelcfg.YOLOMDE_INPUT_SIZE,
        help="Inference input size. Good sizes are multiples of 14, like 196, 224, 280, 336",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=mdecfg.MDE_HALF,
        help="Use float16 depth weights on CUDA for faster inference.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=modelcfg.YOLOMDE_FRAME_STRIDE,
        help="Process every Nth frame.",
    )
    parser.add_argument(
        "--depth-format",
        choices=("npz", "npy"),
        default=mdecfg.MDE_DEPTH_FORMAT,
        help="Raw depth frame format. npy is faster; npz is smaller.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=yolocfg.YOLO_SMOOTHING_ALPHA,
        help="Temporal smoothing factor for YOLO joint xy positions.",
    )
    parser.add_argument(
        "--yolo-device",
        default="cpu",
        help="Device for YOLO pose inference. Defaults to CPU.",
    )
    parser.add_argument(
        "--depth-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for depth inference. Defaults to CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--save-frame-joints",
        action=argparse.BooleanOptionalAction,
        default=modelcfg.YOLOMDE_SAVE_JOINTS,
        help="Save raw joint xy positions for every processed frame.",
    )
    parser.add_argument(
        "--save-frame-depths",
        action=argparse.BooleanOptionalAction,
        default=modelcfg.YOLOMDE_SAVE_DEPTHS,
        help="Save raw metric depth arrays for every processed frame.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1.")
    if args.input_size < 1:
        raise ValueError("--input-size must be at least 1.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1.")
    if not 0.0 <= args.smooth_alpha <= 1.0:
        raise ValueError("--smooth-alpha must be between 0.0 and 1.0.")



def main():
    args = parse_args()
    validate_args(args)





    processes = [
        Process(target=run_depth_capture, args=(args,), name="mde-depth"),
        Process(target=run_yolo_capture, args=(args,), name="yolo-pose"),
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    failed = [process.name for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"Parallel capture failed in: {', '.join(failed)}")


if __name__ == "__main__":
    freeze_support()
    main()
