"""
Docstring for dtu_mlops_111.visualize

Module to visualize images from the processed dataset.

Provides a CLI to display and save images from the dataset.

Example usage:
    uv run ./src/dtu_mlops_111/visualize.py --index 112
"""


from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

app = typer.Typer(add_completion=False)

DEFAULT_PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"
DEFAULT_FIG = Path(__file__).parent.parent.parent / "reports" / "figures" / "first_train_image.png"


def _to_display_image(img_t: torch.Tensor) -> tuple[np.ndarray, dict]:
    """Convert a CHW PyTorch tensor to an HWC NumPy image ready for imshow.

    Args:
        img_t: Image tensor.

    Returns:
        Tuple of (image array in HWC, optional imshow kwargs like cmap).
    """
    img_np = img_t.detach().cpu().numpy()
    cmap_kwargs: dict = {}

    # If CHW, convert to HWC
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
        img_np = np.transpose(img_np, (1, 2, 0))

    # If grayscale, set cmap
    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
        cmap_kwargs = {"cmap": "gray"}

    # Normalize to [0, 1] for matplotlib
    if img_np.dtype.kind == "f" and img_np.max() > 1.5:
        img_np = (img_np / 255.0).clip(0.0, 1.0)
    elif img_np.dtype.kind in ("u", "i"):
        maxv = img_np.max() if img_np.size else 255
        if maxv > 1:
            img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)

    return img_np, cmap_kwargs


@app.command()
def show_patch(
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED, help="Directory with train_images.pt and train_target.pt"),
    save_path: Optional[Path] = typer.Option(None, help="Optional path to save the figure"),
    no_show: bool = typer.Option(False, help="Do not display; only save if save_path is given"),
    index: int = typer.Option(0, help="Index of the image to visualize"),
) -> None:
    """Visualize the first image in train_images.pt and its label."""
    images_path = processed_dir / "train_images.pt"
    targets_path = processed_dir / "train_target.pt"

    if not images_path.exists() or not targets_path.exists():
        typer.echo(f"Missing processed tensors in {processed_dir}. Run preprocessing first.")
        raise typer.Exit(code=1)

    train_images = torch.load(images_path)
    train_target = torch.load(targets_path)

    if index < 0 or index >= len(train_images):
        typer.echo(f"Index {index} out of range. Available indices: 0-{len(train_images)-1}")
        raise typer.Exit(code=1)

    img_t = train_images[index]
    label_t = train_target[index]
    label = int(label_t.item()) if hasattr(label_t, "item") else int(label_t)
    label_name = "Ship" if label == 1 else "No Ship"

    img_np, cmap_kwargs = _to_display_image(img_t)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np, **cmap_kwargs)
    plt.axis("off")
    plt.title(f"Label: {label} ({label_name})")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        typer.echo(f"Saved figure to {save_path}")

    if not no_show:
        plt.show()
    else:
        plt.close()

    typer.echo(f"Displayed image at index {index} with label: {label} ({label_name})")


if __name__ == "__main__":
    app()