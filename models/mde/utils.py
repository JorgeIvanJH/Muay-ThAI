from transformers import pipeline
import config as mdecfg
from huggingface_hub import snapshot_download


def safe_model_load(model_path):
    try:
        model = pipeline("depth-estimation", model=model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Attempting to download the model snapshot...")
        snapshot_download(mdecfg.MDE_HF_REPO_ID, local_dir=mdecfg.MDE_HF_WEIGHTS)
        model = pipeline("depth-estimation", model=mdecfg.MDE_HF_WEIGHTS)
    return model