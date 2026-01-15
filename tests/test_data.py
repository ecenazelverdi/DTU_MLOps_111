import numpy as np
from pathlib import Path

import pytest
from PIL import Image
from dtu_mlops_111 import data


def save_dummy_rgb(path: Path, size=(4, 4), color=(10, 20, 30)):
    """Helper: save a tiny RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def test_rgb_mask_to_class_mask_basic():
    # Take up to 4 known colors from the mapping
    items = list(data.RGB_TO_CLASS.items())
    assert items, "RGB_TO_CLASS must not be empty for this test."
    colors = items[: min(4, len(items))]

    unknown_rgb = (123, 45, 67)

    rgb_arr = np.zeros((3, 3, 3), dtype=np.uint8)

    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (rgb, cls_id), (i, j) in zip(colors, coords):
        rgb_arr[i, j] = np.array(rgb, dtype=np.uint8)

    # Put an unknown RGB in one pixel
    rgb_arr[2, 2] = np.array(unknown_rgb, dtype=np.uint8)

    class_mask = data.rgb_mask_to_class_mask(rgb_arr)

    assert class_mask.shape == (3, 3)
    assert class_mask.dtype == np.uint8

    # Known colors map to their class IDs
    for (rgb, cls_id), (i, j) in zip(colors, coords):
        assert class_mask[i, j] == cls_id

    # Unknown color should be mapped to background (class 0)
    assert class_mask[2, 2] == 0


def test_rgb_mask_to_class_mask_bad_shape_raises():
    # Missing channel dimension → should raise
    arr = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        data.rgb_mask_to_class_mask(arr)


def test_pair_images_and_masks_happy_path(tmp_path):
    imgs = tmp_path / "original_images"
    masks = tmp_path / "label_images_semantic"

    save_dummy_rgb(imgs / "img1.png")
    save_dummy_rgb(imgs / "img2.jpg")
    save_dummy_rgb(masks / "img1.png")
    save_dummy_rgb(masks / "img2.png")

    pairs = data.pair_images_and_masks(imgs, masks)

    assert len(pairs) == 2
    stems = {p.image.stem for p in pairs}
    assert stems == {"img1", "img2"}
    for p in pairs:
        assert p.image.stem == p.mask.stem


def test_pair_images_and_masks_missing_mask_raises(tmp_path):
    imgs = tmp_path / "original_images"
    masks = tmp_path / "label_images_semantic"

    save_dummy_rgb(imgs / "has_mask.png")
    save_dummy_rgb(masks / "has_mask.png")
    save_dummy_rgb(imgs / "no_mask.png")  # no corresponding mask

    with pytest.raises(FileNotFoundError) as excinfo:
        data.pair_images_and_masks(imgs, masks)
    assert "Masks missing for stems" in str(excinfo.value)


def test_mydataset_len_and_pairs(tmp_path):
    root = tmp_path / "data_root"
    imgs = root / "original_images"
    masks = root / "label_images_semantic"

    save_dummy_rgb(imgs / "a.png")
    save_dummy_rgb(imgs / "b.jpg")
    save_dummy_rgb(masks / "a.png")
    save_dummy_rgb(masks / "b.png")

    ds = data.MyDataset(root)
    assert len(ds) == 2
    stems = sorted(p.image.stem for p in ds.pairs)
    assert stems == ["a", "b"]


def test_maybe_resize_pil_no_resize_returns_same_image():
    img = Image.new("RGB", (10, 20), color=(0, 0, 0))
    out = data._maybe_resize_pil(img, resize=None, is_mask=False)
    # For resize=None you return the original object
    assert out is img


def test_maybe_resize_pil_mask_preserves_label_values():
    # 2-class mask: left half 0, right half 1
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[:, 2:] = 1
    mask_img = Image.fromarray(arr, mode="L")

    resized = data._maybe_resize_pil(mask_img, resize=8, is_mask=True)
    arr_resized = np.array(resized)

    assert resized.size == (8, 8)
    # With nearest-neighbor, we should still only see {0, 1}
    unique_vals = set(np.unique(arr_resized).tolist())
    assert unique_vals <= {0, 1}

