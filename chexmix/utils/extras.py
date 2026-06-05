from PIL import Image
import numpy as np
import os
import torch
from safetensors.torch import load_file

SPECIAL_TOKENS = {50295}


def load_pil_image(image_path):
    if image_path.endswith(".png"):
        img = Image.open(image_path)

        arr = np.array(img).astype(np.float32)

        # Percentile windowing (tweak if needed)
        lo, hi = np.percentile(arr, (1, 99))

        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-6)
        arr = (arr * 255).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")
    elif image_path.endswith(".jpg"):
        return Image.open(image_path).convert("RGB")
    else:
        raise ValueError(f"Unsupported image format: {image_path}")


def truncate_at_special(t: torch.Tensor) -> torch.Tensor:
    # Find the first index where t is equal to any special token
    mask = torch.isin(
        t, torch.tensor(list(SPECIAL_TOKENS), device=t.device, dtype=t.dtype)
    )

    if mask.any():
        first_special_idx = mask.nonzero(as_tuple=True)[0][0].item()
        return t[:first_special_idx]
    else:
        return t


def load_checkpoint(checkpoint_path, **kwargs):
    # See if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

    # Load the checkpoint
    checkpoint = load_file(checkpoint_path)
    if kwargs["ImageEmbeddings"]:
        # Load the checkpoint
        new_checkpoint = {}

        # Remove the 'model.' prefix from the keys
        for key, value in checkpoint.items():
            if key.startswith("model."):
                new_checkpoint[key[6:]] = value
        return new_checkpoint
    else:
        return checkpoint
