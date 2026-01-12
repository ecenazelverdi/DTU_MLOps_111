from pathlib import Path
import os
import shutil
import typer
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from dotenv import load_dotenv

load_dotenv()
print(f"Debug: KAGGLE_USERNAME loaded: {bool(os.environ.get('KAGGLE_USERNAME'))}")


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.images = sorted(
            list((data_path / "original_images").glob("*.png")) + list((data_path / "original_images").glob("*.jpg")))
        self.masks = sorted(list((data_path / "label_images_semantic").glob("*.png")) + list(
            (data_path / "label_images_semantic").glob("*.jpg")))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        img_path = self.images[index]
        mask_path = self.masks[index]
        return img_path, mask_path

    def rgb_to_mask(self, rgb_arr):
        """Convert RGB mask to 2D integer mask."""
        mask = np.zeros((rgb_arr.shape[0], rgb_arr.shape[1]), dtype=np.uint8) + 5
        colors = {
            0: [155, 38, 182], 1: [14, 135, 204], 2: [124, 252, 0],
            3: [255, 20, 147], 4: [169, 169, 169]
        }
        for class_id, color in colors.items():
            matches = np.all(rgb_arr == color, axis=-1)
            mask[matches] = class_id
        return mask

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        if len(self.images) == 0:
            print("No images found.")
            return

        # 1. Create output directories for train/val/test splits
        for split in ["train", "val", "test"]:
            (output_folder / split / "images").mkdir(parents=True, exist_ok=True)
            (output_folder / split / "masks").mkdir(parents=True, exist_ok=True)

        # 2. Split the dataset: 80% Train, 10% Validation, 10% Test
        train_imgs, test_imgs, train_masks, test_masks = train_test_split(self.images, self.masks, test_size=0.2,
                                                                          random_state=42)
        val_imgs, test_imgs, val_masks, test_masks = train_test_split(test_imgs, test_masks, test_size=0.5,
                                                                      random_state=42)

        splits = {"train": (train_imgs, train_masks), "val": (val_imgs, val_masks), "test": (test_imgs, test_masks)}

        # 3. Process each split
        for split_name, (imgs, masks) in splits.items():
            print(f"Processing {split_name} split ({len(imgs)} images)...")
            for i, (img_path, mask_path) in enumerate(zip(imgs, masks)):
                img = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
                mask = Image.open(mask_path).convert("RGB").resize((512, 512), Image.NEAREST)

                # Convert RBG mask to Integer mask (Class ID 0-4)
                mask_indices = self.rgb_to_mask(np.array(mask))

                # Save processed files
                img.save(output_folder / split_name / "images" / img_path.name)
                Image.fromarray(mask_indices).save(output_folder / split_name / "masks" / img_path.name)


def preprocess(
        data_path: Path = Path("data/raw/classes_dataset/classes_dataset"),
        output_folder: Path = Path("data/processed")
) -> None:
    print("Preprocessing data...")
    if output_folder.exists() and any(output_folder.iterdir()):
        print(f"Output folder {output_folder} is not empty. Skipping preprocessing.")
        return

    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)
    print("Preprocessing complete.")


def download(
        data_path: Path = Path("data/raw"),
        dataset: str = "santurini/semantic-segmentation-drone-dataset"
) -> None:
    """Download and extract the dataset."""
    raw_dir = data_path

    # Check if dataset already exists to avoid re-downloading
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"Data directory {raw_dir} is not empty. Skipping download.")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset} to {raw_dir}...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=raw_dir, unzip=True)

    # Cleanup unnecessary folders
    unnecessary_folders = ["binary_dataset", "semantic_drone_dataset"]
    for folder_name in unnecessary_folders:
        folder_path = raw_dir / folder_name
        if folder_path.exists() and folder_path.is_dir():
            print(f"Removing unnecessary folder: {folder_path}")
            shutil.rmtree(folder_path)

    print("Download and extraction complete.")


app = typer.Typer()
app.command()(download)
app.command()(preprocess)


@app.command()
def main():
    download()
    preprocess()


if __name__ == "__main__":
    app()
