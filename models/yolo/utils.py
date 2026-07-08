import argparse
import json
import re
from datetime import datetime
from pathlib import Path
import cv2
from ultralytics import YOLO
import config as yolocfg

def boxes_to_detections(result):
    if result.boxes is None:
        return []

    boxes = []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
    cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

    for box_index, coordinates in enumerate(xyxy):
        boxes.append(
            {
                "box_index": box_index,
                "xyxy": [float(value) for value in coordinates],
                "confidence": float(conf[box_index]) if len(conf) > box_index else None,
                "class_id": int(cls[box_index]) if len(cls) > box_index else None,
            }
        )

    return boxes

def keypoints_to_people(result):
    if result.keypoints is None:
        return []

    xy = result.keypoints.xy.cpu().numpy()
    conf = None
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf.cpu().numpy()

    people = []
    for person_index, person in enumerate(xy):
        keypoints = []
        for keypoint_index, coordinates in enumerate(person):
            keypoint_name = (
                yolocfg.YOLO_KEYPOINT_NAMES[keypoint_index]
                if keypoint_index < len(yolocfg.YOLO_KEYPOINT_NAMES)
                else f"keypoint_{keypoint_index}"
            )
            score = None if conf is None else float(conf[person_index][keypoint_index])
            keypoints.append(
                {
                    "name": keypoint_name,
                    "x": float(coordinates[0]),
                    "y": float(coordinates[1]),
                    "confidence": score,
                }
            )

        people.append(
            {
                "person_index": person_index,
                "keypoints": keypoints,
            }
        )

    return people
