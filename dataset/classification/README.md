# Label Studio classification exports

Guard and striking are independent Label Studio projects. Export each project
to its matching folder:

    classification/
      guard/
        <label-studio-export>.json
      striking/
        <label-studio-export>.json

The guard project must contain exactly:

    background
    guard_up
    guard_down

The striking project must contain exactly:

    background
    punch
    elbow
    kick
    knee

Do not combine both vocabularies in one project or export. The dataset builder
validates the complete export before YOLO runs and rejects unexpected or
missing task labels.

Keep one current JSON export in each task folder when relying on automatic
discovery. If the folder contains multiple exports, select one explicitly with
the --annotations argument.

Build the existing guard export:

    python dataset/build_action_joint_dataset.py --task guard

Build striking after its annotations are ready:

    python dataset/build_action_joint_dataset.py --task striking

Both projects must label every frame of every exported video, including
background frames.
