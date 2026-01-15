import os

from invoke import Context, task

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

# ToDo: These two tasks needs to be updated according to the incoming training and evaluation scripts
@task
def train(ctx: Context, lr: float = 1e-4, batch_size: int = 4, epochs: int = 10) -> None:
    """Train model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/train.py --lr {lr} --batch-size {batch_size} --epochs {epochs}", 
        echo=True, 
        pty=not WINDOWS
    )

@task
def evaluate(ctx: Context, checkpoint: str = "latest") -> None:
    """Evaluate model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py --checkpoint {checkpoint}", 
        echo=True, 
        pty=not WINDOWS
    )
#################################################################################

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

# ToDo: These tasks needed to be checked
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
