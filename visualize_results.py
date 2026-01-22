#!/usr/bin/env python3
"""
Visualize nnU-Net segmentation results by overlaying masks on original images.

This script creates side-by-side visualizations showing:
- Original RGB image
- Segmentation mask with color-coded classes
- Overlay of mask on original image

Usage:
    python visualize_results.py images_raw/ nnUNet_results/inference_outputs/ visualizations/
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from loguru import logger
import wandb


# Class colors matching the dataset
CLASS_COLORS = {
    0: (0, 0, 0),  # Background (black)
    1: (155, 38, 182),  # Obstacles (purple)
    2: (14, 135, 204),  # Water (blue)
    3: (124, 252, 0),  # Soft-surfaces (green)
    4: (255, 20, 147),  # Moving-objects (pink)
    5: (169, 169, 169),  # Landing-zones (gray)
}

CLASS_NAMES = {
    0: "Background",
    1: "Obstacles",
    2: "Water",
    3: "Soft-surfaces",
    4: "Moving-objects",
    5: "Landing-zones",
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert grayscale segmentation mask to RGB colored mask.

    Args:
        mask: Grayscale mask with class IDs (H, W)

    Returns:
        RGB colored mask (H, W, 3)
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        colored_mask[mask == class_id] = color

    return colored_mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay colored mask on original image.

    Args:
        image: Original RGB image (H, W, 3)
        mask: Grayscale segmentation mask (H, W)
        alpha: Transparency of mask overlay (0=transparent, 1=opaque)

    Returns:
        Overlaid image (H, W, 3)
    """
    colored_mask = colorize_mask(mask)
    return (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)


def create_visualization(
    original_image: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create visualization showing original, colored mask, and overlay.

    Args:
        original_image: Original RGB image (H, W, 3)
        mask: Grayscale segmentation mask (H, W)

    Returns:
        Tuple of (colored_mask, overlay, combined_viz)
        - colored_mask: RGB colored mask
        - overlay: Mask overlaid on original
        - combined_viz: Side-by-side visualization
    """
    colored_mask = colorize_mask(mask)
    overlaid = overlay_mask(original_image, mask, alpha=0.4)

    # Create side-by-side visualization
    h, w = original_image.shape[:2]
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, :w] = original_image
    combined[:, w : 2 * w] = colored_mask
    combined[:, 2 * w :] = overlaid

    return colored_mask, overlaid, combined


def visualize_results(
    images_dir: Path, masks_dir: Path, output_dir: Path, save_individual: bool = True
):
    """Visualize all segmentation results.

    Args:
        images_dir: Directory with original RGB images
        masks_dir: Directory with segmentation masks (from inference)
        output_dir: Directory to save visualizations
        save_individual: If True, save colored mask and overlay separately
    """
    # Initialize W&B
    if wandb.run is None:
        wandb.init(project="semantic_segmentation_nnunet_inference", job_type="visualization")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of mask files
    mask_files = sorted(masks_dir.glob("case_*.png"))

    if not mask_files:
        logger.error(f"No mask files found in {masks_dir}")
        print(f"❌ No mask files found in {masks_dir}")
        return

    logger.info(f"Found {len(mask_files)} masks for visualization")
    print(f"📁 Found {len(mask_files)} masks")
    
    # Log to W&B
    wandb.log({"visualization/total_masks": len(mask_files)})

    # Map mask files to original images
    image_files = sorted(images_dir.glob("*"))
    image_files = [
        f for f in image_files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ]

    if len(image_files) != len(mask_files):
        print(
            f"⚠️  Warning: Found {len(image_files)} images but {len(mask_files)} masks"
        )

    for i, (mask_path, image_path) in enumerate(zip(mask_files, image_files)):
        print(f"🎨 Processing {i+1}/{len(mask_files)}: {image_path.name}...", end=" ")

        # Load images
        original = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Resize mask to match original if needed
        if original.shape[:2] != mask.shape:
            mask_img = Image.fromarray(mask)
            mask_img = mask_img.resize(
                (original.shape[1], original.shape[0]), Image.NEAREST
            )
            mask = np.array(mask_img)

        # Create visualizations
        colored_mask, overlaid, combined = create_visualization(original, mask)

        # Save combined visualization
        case_id = mask_path.stem  # e.g., "case_0000"
        combined_path = output_dir / f"{case_id}_visualization.png"
        Image.fromarray(combined).save(combined_path)
        
        # Upload to W&B
        if wandb.run and i < 10:  # Upload first 10 to avoid too many images
            wandb.log({
                f"visualizations/{case_id}": wandb.Image(
                    combined,
                    caption=f"{image_path.name}"
                )
            })

        if save_individual:
            # Save colored mask
            mask_colored_path = output_dir / f"{case_id}_colored_mask.png"
            Image.fromarray(colored_mask).save(mask_colored_path)

            # Save overlay
            overlay_path = output_dir / f"{case_id}_overlay.png"
            Image.fromarray(overlaid).save(overlay_path)

        print("✅")

    logger.success(f"Visualizations saved to: {output_dir}")
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print(
        f"📊 Generated {len(mask_files)} visualizations"
        + (f" ({len(mask_files) * 3} files)" if save_individual else "")
    )
    
    # Log summary to W&B
    if wandb.run:
        wandb.log({
            "visualization/completed": len(mask_files),
            "visualization/output_dir": str(output_dir)
        })
        logger.info("Visualization metrics logged to W&B")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize nnU-Net segmentation results"
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory containing original RGB images (e.g., images_raw/)",
    )
    parser.add_argument(
        "masks_dir",
        type=Path,
        help="Directory containing segmentation masks (e.g., nnUNet_results/inference_outputs/)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save visualizations (e.g., visualizations/)",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only save combined visualization (not individual colored mask and overlay)",
    )

    args = parser.parse_args()

    visualize_results(
        args.images_dir, args.masks_dir, args.output_dir, not args.combined_only
    )


if __name__ == "__main__":
    main()
