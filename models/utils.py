import re
from datetime import datetime
from pathlib import Path


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"

def parse_source(source):
    if str(source).isdigit():
        return int(source), f"webcam-{source}"
    path = Path(source)
    return str(path), path.stem

def build_output_paths(output_dir, input_dir, model_path):
    model_label = Path(model_path).stem
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = f"{slugify(input_dir)}__{slugify(model_label)}__{run_id}"
    run_dir = Path(output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{slugify(input_dir)}__{slugify(model_label)}"
    return {
        "run_dir": run_dir,
        "video": run_dir / f"{file_prefix}_annotated.mp4",
        "predictions": run_dir / f"{file_prefix}_predictions.jsonl",
    }



