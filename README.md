Design Choices:

- We assume 1 training session with one person only. Ideally looking sideways and ideally shadowboxing.
- Classification model only on action, not limb (e.g. no left punch, right punch label), leaving YOLO to detect the limb, and the classification only on the action. joint effor of YOLO + classification detects action with correspodnding limb
- Yolo pose detection (model.predict()) is preferred for now over pose tracking (model.track()) because it is faster and we assume the same person is always going to be the one closer to the camera.