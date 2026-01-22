import typer
import os
import wandb


def link_model(artifact_path: str, aliases: list[str] = ["staging"]) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with.

    Example:
        python scripts/link_model.py entity/project/artifact_name:version -a staging -a best

    """
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    # Handle potentially different path formats if needed, but assuming standard wandb format
    try:
        _, _, artifact_name_version = artifact_path.split("/")
        artifact_name, _ = artifact_name_version.split(":")
    except ValueError:
        # Fallback parsing or error if format is unexpected
        typer.echo(f"Could not parse artifact path: {artifact_path}. Expected entity/project/artifact_name:version")
        return

    try:
        artifact = api.artifact(artifact_path)
        # We need to link it to the model registry "collection". 
        # In WandB, the model registry is a special collection type.
        # usually named "model-registry/ModelName"
        
        target_path = f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}"
        
        artifact.link(target_path=target_path, aliases=aliases)
        artifact.save()
        typer.echo(f"Artifact {artifact_path} linked to {target_path} with aliases {aliases}")
    except Exception as e:
        typer.echo(f"Error linking artifact: {e}")
        raise e

if __name__ == "__main__":
    typer.run(link_model)
