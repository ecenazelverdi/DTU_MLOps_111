import shutil
import subprocess
from pathlib import Path

import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


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
        Path(__file__).parent.parent.parent / "data" / "preprocessed",
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
