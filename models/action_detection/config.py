from collections.abc import Iterable
from pathlib import Path


TASK_CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "guard": ("background", "guard_up", "guard_down"),
    "striking": ("background", "punch", "elbow", "kick", "knee"),
}


def class_names_for_task(task: str) -> tuple[str, ...]:
    """
    Return the only labels allowed for an action-classification task.
    e.g:
        "guard" -> ("background", "guard_up", "guard_down")
        "striking" -> ("background", "punch", "elbow", "kick", "knee")
    """

    try:
        return TASK_CLASS_NAMES[task]
    except KeyError as error:
        choices = ", ".join(sorted(TASK_CLASS_NAMES))
        raise ValueError(
            f"Unknown classification task {task!r}; expected one of: {choices}"
        ) from error


def resolve_classification_task(
    task: str | None,
    dataset_dir: Path | str,
) -> str:
    """
    Resolve a task explicitly or from a guard/striking dataset folder.

    Raises ValueError if the task cannot be inferred or if it conflicts with the dataset folder name.
    """

    dataset_task = Path(dataset_dir).name
    if dataset_task not in TASK_CLASS_NAMES:
        dataset_task = ""
    if task is None:
        if dataset_task:
            return dataset_task
        choices = ", ".join(sorted(TASK_CLASS_NAMES))
        raise ValueError(
            "Cannot infer the classification task from dataset directory "
            f"{dataset_dir!s}; pass --task with one of: {choices}"
        )
    class_names_for_task(task)
    if dataset_task and dataset_task != task:
        raise ValueError(
            f"Task {task!r} conflicts with dataset folder {dataset_task!r}"
        )
    return task


def validate_task_labels(
    task: str,
    labels: Iterable[str],
    *,
    source: str,
    require_all: bool = True,
) -> tuple[str, ...]:
    """
    Validate an observed label collection against one task vocabulary.

    Raises ValueError if any label is invalid for the task or if any required label is missing.
    """

    expected = class_names_for_task(task)
    observed = tuple(sorted(set(str(label) for label in labels)))
    unexpected = sorted(set(observed) - set(expected))
    if unexpected:
        raise ValueError(
            f"{source} contains labels invalid for task {task!r}: "
            + ", ".join(unexpected)
            + f". Allowed labels: {', '.join(expected)}"
        )
    missing = sorted(set(expected) - set(observed))
    if require_all and missing:
        raise ValueError(
            f"{source} is missing required labels for task {task!r}: "
            + ", ".join(missing)
        )
    return observed


SKELETON_EDGES = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

CLASS_COLORS = {
    "guard_up": (40, 200, 40),
    "guard_down": (230, 50, 50),
    "background": (150, 150, 150),
    "punch": (50, 100, 240),
    "elbow": (245, 150, 40),
    "kick": (170, 60, 220),
    "knee": (220, 80, 180),
}
