import cv2
from ultralytics import YOLO

# Load the native Pose model (runs incredibly fast, even on a CPU)
model = YOLO('yolov8n-pose.pt')

# Open your boxing video or live webcam stream
cap = cv2.VideoCapture("videos\\Rodtang-taetat-1.mp4") # Replace with 0 for webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference on the current video frame
    results = model(frame, verbose=False)

    for result in results:
        if result.keypoints is not None:
            # keypoints.xy contains the coordinates for all detected people
            # Shape: [num_people, 17 keypoints, 2 coordinates (x,y)]
            keypoints = result.keypoints.xy.cpu().numpy()

            for person in keypoints:
                # Keypoint index references: 
                # 9 = Left Wrist, 10 = Right Wrist, 5 = Left Shoulder, 6 = Right Shoulder
                left_wrist = person[9]
                right_wrist = person[10]

                # Your custom tracking math logic goes here!
                # e.g., Track the frame-by-frame coordinate changes of right_wrist[0] 
                # to calculate the velocity of a straight cross punch.

    # Display the live tracked skeleton skeleton overlay
    annotated_frame = results[0].plot()
    cv2.imshow("AI Combat Coach", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
