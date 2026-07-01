from pathlib import Path

import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig

repo_root = Path(__file__).resolve().parent
depth_pro_root = repo_root / "models" / "depth" / "ml-depth-pro"

image_path = depth_pro_root / "data" / "example.jpg"
checkpoint_path = depth_pro_root / "checkpoints" / "depth_pro.pt"

if not checkpoint_path.is_file():
    raise FileNotFoundError(
        f"Depth Pro checkpoint not found at {checkpoint_path}. "
        "Run get_pretrained_models.sh from models/depth/ml-depth-pro first."
    )

config = DepthProConfig(
    patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
    image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
    decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
    checkpoint_uri=str(checkpoint_path),
    fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
    use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
)

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(config=config)
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

print(f"Depth map shape: {tuple(depth.shape)}")
print(f"Focal length: {focallength_px}")
