from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import typer
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

app = typer.Typer(add_completion=False)

# -----------------------------------------------------------------------------
# Semantic label definition
# -----------------------------------------------------------------------------
# nnU-Net expects background to be encoded as 0. Foreground classes must be
# positive integers.
LABELS: Dict[int, str] = {
    0: "background",
    1: "obstacles",
    2: "water",
    3: "soft_surfaces",
    4: "moving_objects",
    5: "landing_zones",
}

# RGB values used in the raw semantic masks (Kaggle dataset).
# Any pixel not matching a known RGB triplet is mapped to background (0).
RGB_TO_CLASS: Dict[Tuple[int, int, int], int] = {
    (155, 38, 182): 1,  # obstacles
    (14, 135, 204): 2,  # water
    (124, 252, 0): 3,   # soft_surfaces
    (255, 20, 147): 4,  # moving_objects
    (169, 169, 169): 5, # landing_zones
}


# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------
def _find_images(folder: Path) -> List[Path]:
    return sorted([*folder.glob("*.png"), *folder.glob("*.jpg"), *folder.glob("*.jpeg")])


def _stem(p: Path) -> str:
    return p.stem


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _maybe_resize_pil(img: Image.Image, resize: Optional[int], is_mask: bool) -> Image.Image:
    """
    Optional square resize. Intended only for quick experiments / debugging.
    Note: resizing to NxN changes aspect ratio.
    """
    if resize is None:
        return img
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((resize, resize), resample=resample)


def _save_png_l(arr2d: np.ndarray, out_path: Path) -> None:
    """Write a 2D uint8 array as grayscale PNG (mode 'L')."""
    Image.fromarray(arr2d.astype(np.uint8), mode="L").save(out_path)


# -----------------------------------------------------------------------------
# Mask conversion
# -----------------------------------------------------------------------------
def rgb_mask_to_class_mask(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Convert an RGB semantic mask (H, W, 3) to a class-index mask (H, W) encoded
    as uint8. Unknown colors are assigned to background (0).
    """
    if rgb_arr.ndim != 3 or rgb_arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB array (H,W,3), got {rgb_arr.shape}")

    h, w, _ = rgb_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Assign known colors; everything else stays 0.
    for rgb, cls_id in RGB_TO_CLASS.items():
        matches = np.all(rgb_arr == np.array(rgb, dtype=np.uint8), axis=-1)
        mask[matches] = cls_id

    return mask


def _rgb_unknown_mask(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask (H, W) indicating pixels whose RGB value is not part
    of the known color map.
    """
    if rgb_arr.ndim != 3 or rgb_arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB array (H,W,3), got {rgb_arr.shape}")

    known = np.zeros(rgb_arr.shape[:2], dtype=bool)
    for rgb in RGB_TO_CLASS.keys():
        known |= np.all(rgb_arr == np.array(rgb, dtype=np.uint8), axis=-1)
    return ~known


# -----------------------------------------------------------------------------
# Pairing logic
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Pair:
    image: Path
    mask: Path


def pair_images_and_masks(images_dir: Path, masks_dir: Path) -> List[Pair]:
    """
    Pair images and masks by filename stem (e.g., 192.jpg <-> 192.png).
    Fails fast on missing counterparts to avoid silent dataset corruption.
    """
    images = _find_images(images_dir)
    masks = _find_images(masks_dir)

    img_map = {_stem(p): p for p in images}
    mask_map = {_stem(p): p for p in masks}

    common = sorted(set(img_map.keys()) & set(mask_map.keys()))
    missing_masks = sorted(set(img_map.keys()) - set(mask_map.keys()))
    missing_imgs = sorted(set(mask_map.keys()) - set(img_map.keys()))

    if missing_masks:
        raise FileNotFoundError(
            f"Masks missing for stems: {missing_masks[:10]}{'...' if len(missing_masks) > 10 else ''}"
        )
    if missing_imgs:
        raise FileNotFoundError(
            f"Images missing for stems: {missing_imgs[:10]}{'...' if len(missing_imgs) > 10 else ''}"
        )

    return [Pair(img_map[s], mask_map[s]) for s in common]


# -----------------------------------------------------------------------------
# Dataset container (no torch dependency; nnU-Net workflow uses filesystem)
# -----------------------------------------------------------------------------
class MyDataset:
    """Lightweight container over (image_path, mask_path) pairs."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.images_dir = data_path / "original_images"
        self.masks_dir = data_path / "label_images_semantic"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Missing masks dir: {self.masks_dir}")

        self.pairs = pair_images_and_masks(self.images_dir, self.masks_dir)

        # nnU-Net case_id uses the stem; stems must be unique to avoid collisions.
        stems = [pair.image.stem for pair in self.pairs]
        if len(stems) != len(set(stems)):
            seen = set()
            dupes = []
            for s in stems:
                if s in seen:
                    dupes.append(s)
                seen.add(s)
            raise RuntimeError(
                f"Duplicate image stems found (case_id collision). Examples: {sorted(set(dupes))[:10]}"
            )

    def __len__(self) -> int:
        return len(self.pairs)


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------
@app.command()
def download(
    data_path: Path = typer.Option(Path("data/raw"), help="Download target directory."),
    dataset: str = typer.Option("santurini/semantic-segmentation-drone-dataset", help="Kaggle dataset slug."),
    unzip: bool = typer.Option(True, help="Extract after download."),
    cleanup_zip: bool = typer.Option(True, help="Remove zip after extraction."),
    force: bool = typer.Option(False, help="Overwrite non-empty target directory."),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show progress feedback during download.",
    ),
) -> None:
    """
    Download the Kaggle dataset using kaggle.json credentials (or environment variables).
    """
    typer.echo(f"Debug: KAGGLE_USERNAME loaded: {bool(os.environ.get('KAGGLE_USERNAME'))}")

    raw_dir = data_path
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        typer.echo(f"Data directory {raw_dir} is not empty. Skipping download. Use --force to override.")
        return

    _ensure_dir(raw_dir)

    typer.echo(f"Downloading {dataset} to {raw_dir}...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    if progress:
        typer.echo("(This may take several minutes...)")
    api.dataset_download_files(dataset, path=raw_dir, unzip=unzip)
    
    if progress:
        typer.echo("Download finished. Processing files...")

    zip_name = dataset.split("/")[-1] + ".zip"
    zip_path = raw_dir / zip_name
    if cleanup_zip and zip_path.exists():
        try:
            zip_path.unlink()
            if progress:
                typer.echo("Removed zip archive.")
        except Exception:
            pass

    if unzip:
        unnecessary_folders = ["binary_dataset", "semantic_drone_dataset"]
        folders_to_remove = [
            raw_dir / folder_name
            for folder_name in unnecessary_folders
            if (raw_dir / folder_name).exists() and (raw_dir / folder_name).is_dir()
        ]
        
        if folders_to_remove:
            typer.echo(f"Cleaning up {len(folders_to_remove)} unnecessary folder(s)...")
            for folder_path in tqdm(folders_to_remove, desc="cleanup", unit="folder", disable=not progress):
                shutil.rmtree(folder_path)

    typer.echo("Download complete.")


@app.command("nnunet-export")
def nnunet_export(
    data_path: Path = typer.Option(
        Path("data/raw/classes_dataset/classes_dataset"), help="Extracted dataset root."
    ),
    nnunet_raw_dir: Path = typer.Option(Path("nnUNet_raw"), help="nnUNet_raw root directory."),
    dataset_id: int = typer.Option(101, help="nnU-Net dataset id (DatasetXXX_*)."),
    dataset_name: str = typer.Option("DroneSeg", help="nnU-Net dataset name (DatasetXXX_Name)."),
    seed: int = typer.Option(42, help="Random seed (only used if test_ratio>0)."),
    test_ratio: float = typer.Option(0.0, help="Optional holdout into imagesTs (no labels)."),
    resize: Optional[int] = typer.Option(None, help="Optional square resize (debug)."),
    force: bool = typer.Option(False, help="Overwrite existing nnU-Net dataset folder."),
    warn_unknown_colors: bool = typer.Option(True, help="Warn on unknown RGB in masks."),
    fail_on_unknown_colors: bool = typer.Option(False, help="Fail on unknown RGB ratio over threshold."),
    unknown_color_warn_threshold: float = typer.Option(
        0.001, help="Unknown RGB pixel ratio threshold."
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show progress bars.",
    ),
    converted_by: str = typer.Option("Elif", help="dataset.json field: converted_by."),
    licence: str = typer.Option("unknown", help="dataset.json field: licence."),
    reference: str = typer.Option(
        "santurini/semantic-segmentation-drone-dataset", help="dataset.json field: reference."
    ),
) -> None:
    """
    Export to nnU-Net v2 raw format (2D natural images):
      - imagesTr: case_<id>_0000.png, _0001.png, _0002.png
      - labelsTr: case_<id>.png (uint8 class indices)
      - imagesTs: optional unlabeled holdout set (if test_ratio>0)
      - dataset.json: metadata and label mapping
    """
    ds = MyDataset(data_path)

    if test_ratio < 0 or test_ratio >= 1.0:
        raise ValueError("test_ratio must be in [0, 1). Use 0 for no holdout set.")

    pairs = ds.pairs
    if len(pairs) == 0:
        raise RuntimeError("No pairs found.")

    if test_ratio > 0:
        train_pairs, test_pairs = train_test_split(pairs, test_size=test_ratio, random_state=seed)
    else:
        train_pairs, test_pairs = pairs, []

    dataset_folder = nnunet_raw_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    imagesTr = dataset_folder / "imagesTr"
    labelsTr = dataset_folder / "labelsTr"

    if dataset_folder.exists() and force:
        shutil.rmtree(dataset_folder)
    if dataset_folder.exists() and not force:
        raise FileExistsError(f"{dataset_folder} exists. Use --force to overwrite.")

    _ensure_dir(imagesTr)
    _ensure_dir(labelsTr)

    imagesTs: Optional[Path] = None
    if test_ratio > 0:
        imagesTs = dataset_folder / "imagesTs"
        _ensure_dir(imagesTs)

    file_ending = ".png"

    def export_split(pairs_split: List[Pair], is_train: bool) -> None:
        out_img_dir = imagesTr if is_train else imagesTs
        if out_img_dir is None:
            raise RuntimeError("Internal error: requested test export but imagesTs not created.")

        desc = "export-train" if is_train else "export-test"
        iterable = tqdm(pairs_split, desc=desc, unit="case") if progress else pairs_split

        for pair in iterable:
            case_id = f"case_{pair.image.stem}"

            # Image: exported for both train/test (as 3 modalities: R, G, B).
            img = Image.open(pair.image).convert("RGB")
            img = _maybe_resize_pil(img, resize=resize, is_mask=False)
            img_np = np.array(img, dtype=np.uint8)

            _save_png_l(img_np[:, :, 0], out_img_dir / f"{case_id}_0000{file_ending}")
            _save_png_l(img_np[:, :, 1], out_img_dir / f"{case_id}_0001{file_ending}")
            _save_png_l(img_np[:, :, 2], out_img_dir / f"{case_id}_0002{file_ending}")

            if not is_train:
                continue

            # Label: only for training set.
            mask_rgb = Image.open(pair.mask).convert("RGB")
            mask_rgb = _maybe_resize_pil(mask_rgb, resize=resize, is_mask=True)
            mask_rgb_np = np.array(mask_rgb, dtype=np.uint8)

            if warn_unknown_colors or fail_on_unknown_colors:
                unk_ratio = float(_rgb_unknown_mask(mask_rgb_np).mean())
                if unk_ratio > unknown_color_warn_threshold:
                    msg = (
                        f"Unknown RGB in {pair.mask.name}: "
                        f"{unk_ratio * 100:.3f}% pixels mapped to background."
                    )
                    if fail_on_unknown_colors:
                        raise RuntimeError(f"[FAIL] {msg}")
                    if warn_unknown_colors:
                        typer.echo(f"[WARN] {msg}")

            mask_np = rgb_mask_to_class_mask(mask_rgb_np)
            _save_png_l(mask_np, labelsTr / f"{case_id}{file_ending}")

    typer.echo(f"Exporting nnU-Net dataset to: {dataset_folder}")
    typer.echo(f"Format: {file_ending} | Train: {len(train_pairs)} | Test: {len(test_pairs)}")

    export_split(train_pairs, is_train=True)
    if test_ratio > 0:
        export_split(test_pairs, is_train=False)

    dataset_json = {
        "name": dataset_name,
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {name: idx for idx, name in LABELS.items()},
        "numTraining": len(train_pairs),
        "file_ending": file_ending,
        "licence": licence,
        "converted_by": converted_by,
        "reference": reference,
    }

    with open(dataset_folder / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    typer.echo("nnU-Net export complete.")
    typer.echo(f"Dataset folder: {dataset_folder}")
    typer.echo("\nNext steps (nnU-Net v2):")
    typer.echo(f"  export nnUNet_raw='{nnunet_raw_dir.absolute()}'")
    typer.echo("  export nnUNet_preprocessed='.../nnUNet_preprocessed'")
    typer.echo("  export nnUNet_results='.../nnUNet_results'")
    typer.echo(f"  nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
    typer.echo(f"  nnUNetv2_train {dataset_id} 2d 0")


@app.command()
def main() -> None:
    """Convenience entrypoint: download + nnunet_export."""
    download()
    nnunet_export()


if __name__ == "__main__":
    app()
