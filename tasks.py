import os

from dotenv import load_dotenv
from invoke import Context, task
import json
from pathlib import Path

load_dotenv()

WINDOWS = os.name == "nt"
PROJECT_NAME = "dtu_mlops_111"
PYTHON_VERSION = "3.12"


# Project commands
@task
def download_and_export_data(ctx: Context) -> None:
    """Download and export the data to nnU-Net raw format."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py main", echo=True, pty=not WINDOWS)


@task
def download_data(ctx: Context) -> None:
    """Download data from Kaggle."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py download", echo=True, pty=not WINDOWS)


@task
def export_data(ctx: Context) -> None:
    """Export data to nnU-Net raw format."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py nnunet-export", echo=True, pty=not WINDOWS)


@task
def download_models(ctx: Context) -> None:
    """Download models from DVC."""
    ctx.run("uv run dvc pull", echo=True, pty=not WINDOWS)


# To run APIs
@task
def app(ctx: Context) -> None:
    """Run the API."""
    ctx.run("uv run uvicorn main:app --port 8000 --reload", echo=True, pty=not WINDOWS)


@task
def preprocess(ctx: Context, dataset_id: int = 101) -> None:
    """Run nnU-Net preprocessing."""
    ctx.run(
        f"uv run nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity", echo=True, pty=not WINDOWS
    )


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


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)




# Docker build and run tasks
@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build all docker images (train, inference, api, bento)."""
    print("Building train:latest...")
    ctx.run(
        f"docker build -f train.dockerfile -t train:latest . --progress={progress}",
        echo=True,
        pty=not WINDOWS,
        warn=True,
    )
    print("\nBuilding inference:latest...")
    ctx.run(
        f"docker build -f inference.dockerfile -t inference:latest . --progress={progress}",
        echo=True,
        pty=not WINDOWS,
        warn=True,
    )
    print("\nBuilding api:latest...")
    ctx.run(
        f"docker build -f api.dockerfile -t api:latest . --progress={progress}", echo=True, pty=not WINDOWS, warn=True
    )
    print("\nBuilding bento:latest...")
    ctx.run(
        f"docker build -t {tag} -f bento.dockerfile . --progress={progress}",
        echo=True,
        pty=not WINDOWS,
        warn=True,
    )


@task
def docker_build_train(ctx: Context, progress: str = "plain") -> None:
    """Build training docker image."""
    ctx.run(f"docker build -f train.dockerfile -t train:latest . --progress={progress}", echo=True, pty=not WINDOWS)


@task
def docker_build_api(ctx: Context, progress: str = "plain") -> None:
    """Build API docker image."""
    ctx.run(f"docker build -f api.dockerfile -t api:latest . --progress={progress}", echo=True, pty=not WINDOWS)


@task
def docker_build_inference(ctx: Context, progress: str = "plain") -> None:
    """Build inference docker image."""
    ctx.run(
        f"docker build -f inference.dockerfile -t inference:latest . --progress={progress}", echo=True, pty=not WINDOWS
    )


@task
def docker_train(ctx: Context) -> None:
    """Run training inside Docker container."""
    ctx.run(
        "docker run --gpus all --ipc=host "
        "--env-file .env "
        "-v $(pwd)/data:/app/data "
        "-v $(pwd)/images_raw:/app/images_raw "
        "-v $(pwd)/nnUNet_raw:/app/nnUNet_raw "
        "-v $(pwd)/nnUNet_preprocessed:/app/nnUNet_preprocessed "
        "-v $(pwd)/nnUNet_results:/app/nnUNet_results "
        "train:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_inference(ctx: Context) -> None:
    """Run inference inside Docker container."""
    ctx.run(
        "docker run --gpus all --ipc=host "
        "--env-file .env "
        "-v $(pwd)/nnUNet_results:/nnUnet_results "
        "-v $(pwd)/images_raw:/images_raw "
        "-v $(pwd)/visualizations:/visualizations "
        "inference:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_run_api(ctx: Context) -> None:
    """Run docker container (api:latest)."""
    auth_flags = ""
    # Check if credentials are already set in env (e.g. through .env file)
    gcloud_creds = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(gcloud_creds):
        print(f"No credentials in env. Mounting local GCloud ADC from {gcloud_creds}")
        auth_flags = f"-v {gcloud_creds}:/tmp/gcp_creds.json " "-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_creds.json"
    else:
        print("Warning: No Google credentials found in env or local config. DVC pull might fail.")

    ctx.run(f"docker run -p 8080:8080 --env-file .env {auth_flags} api:latest", echo=True, pty=not WINDOWS)


# ToDo: These tasks needed to be checked

@task
def bento_build(ctx: Context) -> None:
    ctx.run("uv run bentoml build", echo=True, pty=not WINDOWS)


@task
def bento_serve(ctx: Context, port: int = 3000) -> None:
    ctx.run(
        f"uv run bentoml serve {PROJECT_NAME}.bento_service:DroneSegService "
        f"--reload --host 0.0.0.0 --port {port}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_bento(ctx: Context, tag: str = "bento:latest", progress: str = "plain") -> None:
    ctx.run(
        f"docker build -t {tag} -f bento.dockerfile . --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_run_bento(
    ctx: Context,
    tag: str = "bento:latest",
    port: int = 8080,
    enable_dvc_pull: bool = True,
    dvc_strict: bool = False,
) -> None:
    env_file = "--env-file .env" if Path(".env").exists() else ""
    enable = "1" if enable_dvc_pull else "0"
    strict = "1" if dvc_strict else "0"

    auth_flags = ""
    # Check if credentials are already set in env (e.g. through .env file)
    gcloud_creds = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(gcloud_creds):
        print(f"No credentials in env. Mounting local GCloud ADC from {gcloud_creds}")
        auth_flags = f"-v {gcloud_creds}:/tmp/gcp_creds.json " "-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_creds.json"
    else:
        print("Warning: No Google credentials found in env or local config. DVC pull might fail.")

    ctx.run(
        "docker run --rm "
        f"-p {port}:8080 "
        f"{env_file} "
        f"{auth_flags} "
        f"-e PORT=8080 "
        f"-e ENABLE_DVC_PULL={enable} "
        f"-e DVC_STRICT={strict} "
        f"-e nnUNet_raw=/app/nnUNet_raw "
        f"-e nnUNet_preprocessed=/app/nnUNet_preprocessed "
        f"-e nnUNet_results=/app/nnUNet_results "
        f"{tag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def cloud_run_update_env(ctx: Context, service: str = "drone-seg", region: str = "europe-north1") -> None:
    ctx.run(
        f"gcloud run services update {service} --region {region} "
        f'--set-env-vars ENABLE_DVC_PULL=1',
        echo=True,
        pty=not WINDOWS,
    )


@task
def cloud_run_update_resources(
    ctx: Context,
    service: str = "drone-seg",
    region: str = "europe-north1",
    timeout: int = 900,
    cpu: int = 4,
    memory: str = "8Gi",
    min_instances: int = 1,
) -> None:
    ctx.run(
        f"gcloud run services update {service} --region {region} "
        f"--timeout={timeout} --cpu={cpu} --memory={memory} --min-instances={min_instances}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def cloud_run_logs(ctx: Context, service: str = "drone-seg", region: str = "europe-north1", limit: int = 200) -> None:
    ctx.run(
        f"gcloud run services logs read {service} --region {region} --limit {limit}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def bento_smoke_test(ctx: Context, service_url: str = "") -> None:
    url = service_url or os.getenv("SERVICE_URL", "").strip()
    if not url:
        raise RuntimeError("SERVICE_URL is not set. Pass --service-url or set SERVICE_URL in env.")

    ctx.run(f'curl -i "{url}/livez"', echo=True, pty=not WINDOWS)
    ctx.run(f'curl -i "{url}/readyz"', echo=True, pty=not WINDOWS)
    ctx.run(f'curl -i "{url}/docs.json"', echo=True, pty=not WINDOWS)
    ctx.run(
        f'curl -i -X POST "{url}/model_info" -H "Content-Type: application/json" -d "{{}}"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def bento_predict_base64(
    ctx: Context,
    image_path: str,
    service_url: str = "",
    content_type: str = "image/jpeg",
    out_json: str = "resp.json",
) -> None:
    url = service_url or os.getenv("SERVICE_URL", "").strip()
    if not url:
        raise RuntimeError("SERVICE_URL is not set. Pass --service-url or set SERVICE_URL in env.")

    payload_path = Path("payload.json")

    # Create payload.json locally using python (keeps task cross-platform)
    py = f"""
import base64, json
p = r"{image_path}"
b64 = base64.b64encode(open(p, "rb").read()).decode()
payload = {{"req": {{"image_b64": b64, "content_type": "{content_type}"}}}}
json.dump(payload, open("{payload_path.as_posix()}", "w"))
print("saved {payload_path.as_posix()}")
"""
    ctx.run(f"python -c {json.dumps(py)}", echo=True, pty=not WINDOWS)

    ctx.run(
        f'curl -s -o "{out_json}" -w "\\nHTTP %{{http_code}}\\n" '
        f'-X POST "{url}/predict_base64" '
        f'-H "Content-Type: application/json" '
        f'--data-binary @"{payload_path.as_posix()}"',
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --clean", echo=True, pty=not WINDOWS)



@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def publish_docs(ctx: Context) -> None:
    """Build and publish documentation to GitHub Pages."""
    ctx.run("uv run mkdocs gh-deploy --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)




#################################################################################
