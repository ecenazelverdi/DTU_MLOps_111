from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import typer
from PIL import Image

app = typer.Typer(add_completion=False)

# Color mapping for segmentation masks produced by the nnU-Net export.
# Class ids and colors are kept consistent with data.py to avoid confusion
# when visually inspecting labels or model outputs.
CLASS_TO_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # background
    1: (155, 38, 182),  # obstacles
    2: (14, 135, 204),  # water
    3: (124, 252, 0),  # soft_surfaces
    4: (255, 20, 147),  # moving_objects
    5: (169, 169, 169),  # landing_zones
}


def _colorize_mask_array(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2D class-id mask into an RGB image for visualization.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask (H,W), got shape={mask.shape}")

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Warn if something unexpected shows up in the mask
    unknown = sorted(set(np.unique(mask)) - set(CLASS_TO_RGB.keys()))
    if unknown:
        typer.echo(f"[WARN] Found unexpected class ids in mask: {unknown}")

    for class_id, color in CLASS_TO_RGB.items():
        rgb[mask == class_id] = np.array(color, dtype=np.uint8)

    return rgb


@app.command()
def colorize(
    in_mask: Path = typer.Option(
        ...,
        "--in-mask",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to a class-id mask PNG.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        file_okay=True,
        dir_okay=False,
        help="Where to save the colorized mask.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output file if it exists."),
) -> None:
    """
    Colorize a segmentation mask where pixel values represent class ids.

    This is meant as a quick sanity check to visually verify that labels
    look reasonable after preprocessing or model inference.
    """
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out}. Use --overwrite to replace it.")

    out.parent.mkdir(parents=True, exist_ok=True)

    mask = np.array(Image.open(in_mask))

    # If the input has multiple channels, it's likely not a class-id mask
    if mask.ndim == 3:
        raise ValueError(f"Expected a single-channel class-id mask, got shape={mask.shape} instead.")

    mask = mask.astype(np.uint8, copy=False)
    rgb = _colorize_mask_array(mask)

    Image.fromarray(rgb, mode="RGB").save(out)
    typer.echo(f"Saved colorized mask to: {out}")


if __name__ == "__main__":
    app()
