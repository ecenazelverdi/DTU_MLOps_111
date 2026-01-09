import json 
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

import typer
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        # TODO

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path, test_size: float = 0.2) -> None:
        """Preprocess the raw data and save it to the output folder."""
        with open(self.data_path / "shipsnet.json") as data_file:
            dataset = json.load(data_file)
        Shipsnet= pd.DataFrame(dataset)
        print(Shipsnet.head())
        print('')    
        x = np.array(dataset['data']).astype('uint8')
        y = np.array(dataset['labels']).astype('uint8')
        def describeData(a,b):
            print('Total number of images: {}'.format(len(a)))
            print('Number of NoShip Images: {}'.format(np.sum(b==0)))
            print('Number of Ship Images: {}'.format(np.sum(b==1)))
            print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))
            print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))
        describeData(x,y)

        x = x.reshape(-1, 3, 80, 80)  # reshape to (num_samples, channels, height, width)

        # normalization
        x = x.astype('float32')

        # to torch tensors, convert to channel-first (CHW)
        x_tensor = torch.from_numpy(x)# .permute(0, 1, 2, 3).contiguous()  # (N, 3, 80, 80)
        y_tensor = torch.from_numpy(y).long()

        # Split into train and test
        n_samples = len(x_tensor)
        n_test = int(np.ceil(n_samples * test_size))
        indices = np.random.permutation(n_samples)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_images = x_tensor[train_indices]
        train_target = y_tensor[train_indices]
        test_images = x_tensor[test_indices]
        test_target = y_tensor[test_indices]

        # Save to .pt files
        torch.save(train_images, output_folder / "train_images.pt")
        torch.save(train_target, output_folder / "train_target.pt")
        torch.save(test_images, output_folder / "test_images.pt")
        torch.save(test_target, output_folder / "test_target.pt")

        typer.echo(f"Preprocessed data saved to {output_folder}")
        typer.echo(
            f"Train set: {len(train_images)} images, "
            f"Test set: {len(test_images)} images"
        )

        # print dimensions of each
        typer.echo(f"Train images shape: {train_images.shape}")
        typer.echo(f"Train target shape: {train_target.shape}")
        typer.echo(f"Test images shape: {test_images.shape}")
        typer.echo(f"Test target shape: {test_target.shape}")





def download_data(raw_data_path: Path = Path(__file__).parent.parent.parent / "data" / "raw") -> None:
    """Download the ships-in-satellite-imagery dataset from Kaggle.

    Args:
        raw_data_path: Path to save the raw data.

    Raises:
        FileNotFoundError: If Kaggle API credentials are not configured.
        RuntimeError: If the Kaggle API call fails.
    """
    raw_data_path.mkdir(parents=True, exist_ok=True)

    kaggle_dataset = "rhammell/ships-in-satellite-imagery"

    try:
        typer.echo(f"Downloading {kaggle_dataset} from Kaggle...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(raw_data_path), "--unzip"],
            check=True,
            capture_output=True,
        )
        typer.echo(f"Successfully downloaded dataset to {raw_data_path}")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error downloading dataset: {e.stderr.decode()}", err=True)
        raise RuntimeError(f"Failed to download Kaggle dataset: {e}")
    except FileNotFoundError:
        typer.echo("Kaggle CLI not found. Please install it with: pip install kaggle", err=True)
        raise


def preprocess(
    data_path: Path = typer.Option(
        Path(__file__).parent.parent.parent / "data" / "raw",
        help="Path to the raw data directory",
    ),
    output_folder: Path = typer.Option(
        Path(__file__).parent.parent.parent / "data" / "processed",
        help="Path to save preprocessed data",
    ),
    download: bool = typer.Option(
        False,
        help="Download the dataset from Kaggle before preprocessing",
    ),
) -> None:
    """Preprocess data from raw to processed format.

    Args:
        data_path: Path to the raw data directory.
        output_folder: Path to save preprocessed data.
        download: Whether to download the dataset from Kaggle.
    """
    if download:
        download_data(data_path)

    typer.echo("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)




if __name__ == "__main__":
    typer.run(preprocess)
