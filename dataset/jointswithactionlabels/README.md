here we store raw YOLO's tracked joints along with frame labels

in this dataset we deal with groups. meaning that joints and classes for each frame (row) should not be sepparated from training/validation/test splits, therefore we also acc

COLUMNS:
video_id
frame_index
people_detected
boxes_detected
detection_index
person_index: From YOLO
box_index: From YOLO
frame_width_px: From YOLO
frame_height_px: From YOLO
bbox_*_px: From YOLO
bbox_confidence: From YOLO
bbox_class_id: From YOLO
joint_*_px: From YOLO
joint_*_confidence: From YOLO
action_label: MANUAL LABEL FROM LABEL STUDIO