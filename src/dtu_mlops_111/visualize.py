import typer
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def colorize_mask(mask_path: Path, output_path: Path):
    """
    Colorizes a segmentation mask where pixel values represent class IDs.
    """
    if not mask_path.exists():
        print(f"Error: Mask not found at {mask_path}")
        return

    # Load mask
    mask = np.array(Image.open(mask_path))

    # Define colors (same as in data.py)
    # 0: obstacles (Purple), 1: water (Blue), 2: soft-surfaces (Green)
    # 3: moving-objects (Pink), 4: landing-zones (Grey), 5: background (Black)
    colors = {
        0: [155, 38, 182],  # obstacles
        1: [14, 135, 204],  # water
        2: [124, 252, 0],  # soft-surfaces
        3: [255, 20, 147],  # moving-objects
        4: [169, 169, 169],  # landing-zones
        5: [0, 0, 0]  # background
    }

    # Create RGB image
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        rgb_image[mask == class_id] = color

    # Save
    Image.fromarray(rgb_image).save(output_path)
    print(f"Saved colorized mask to {output_path}")


if __name__ == "__main__":
    typer.run(colorize_mask)
