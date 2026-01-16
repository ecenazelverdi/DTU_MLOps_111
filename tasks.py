import os
from dotenv import load_dotenv
from invoke import Context, task

load_dotenv()

WINDOWS = os.name == "nt"
PROJECT_NAME = "dtu_mlops_111"
PYTHON_VERSION = "3.12"

# Project commands
@task
def download_data(ctx: Context) -> None:
    """Download data from Kaggle."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py download", echo=True, pty=not WINDOWS)

@task
def export_data(ctx: Context) -> None:
    """Export data to nnU-Net raw format."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py nnunet-export", echo=True, pty=not WINDOWS)
# To run APIs
@task
def app(ctx: Context) -> None:
    """Run the API."""
    ctx.run("uv run uvicorn main:app --port 8000 --reload", echo=True, pty=not WINDOWS)

@task
def preprocess(ctx: Context, dataset_id: int = 101) -> None:
    """Run nnU-Net preprocessing."""
    ctx.run(f"uv run nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context, dataset_id: int = 101, fold: int = 0, dim: str = "2d", device: str = "auto") -> None:
    """
    Run nnU-Net training.
    
    Args:
        dataset_id: Dataset ID.
        fold: Fold number (0-4).
        dim: Dimension (2d or 3d_fullres).
        device: 'auto', 'cuda', 'mps', or 'cpu'.
    """
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    ctx.run(f"uv run nnUNetv2_train {dataset_id} {dim} {fold} -device {device}", echo=True, pty=not WINDOWS)

# ToDo: These tasks needed to be checked
@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
#################################################################################
