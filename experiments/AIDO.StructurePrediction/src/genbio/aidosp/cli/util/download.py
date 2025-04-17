import subprocess
import typing as t
from importlib.resources import as_file, files
from pathlib import Path

import click
import huggingface_hub as hf
from huggingface_hub.errors import RepositoryNotFoundError
from loguru import logger

ParamTypeValue = t.TypeVar("ParamTypeValue")

ALLOWED_MODEL_VERSIONS = ("v0.1.2",)
ALLOWED_DATABASES = ("colabfold_envdb", "uniref30")
MODEL_VERSION_TO_NAME: dict[str, str] = {"v0.1.2": "fold49-v0.1.2.pt"}
DATABASE_TO_DOWNLOAD_SCRIPT: dict[str, str] = {
    "colabfold_envdb": "download_colabfold_envdb.sh",
    "uniref30": "download_uniref30.sh",
}


Database = click.Choice(ALLOWED_DATABASES)
ModelVersion = click.Choice(ALLOWED_MODEL_VERSIONS)


def download_model(output_dir: Path, model_version: str, repo_id: str) -> None:
    logger.info(f"Downloading model version {model_version} to {output_dir}")
    try:
        destination = hf.hf_hub_download(
            repo_id=repo_id,
            filename=MODEL_VERSION_TO_NAME[model_version],
            local_dir=output_dir,
        )
        logger.debug(f"Downloaded to {destination}")
    except RepositoryNotFoundError as e:
        logger.error(
            f"Could not find {MODEL_VERSION_TO_NAME[model_version]} in repository {repo_id}"
        )
        raise e


def download_ccd(output_dir: Path, repo_id: str) -> None:
    logger.info(f"Downloading CCD components to {output_dir}")
    for file in ["components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl"]:
        try:
            destination = hf.hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=output_dir,
            )
            logger.debug(f"Downloaded {file} to {destination}")
        except RepositoryNotFoundError as e:
            logger.error(f"Could not find {file} in repository {repo_id}")
            raise e


# We will just call the download script from subprocess here
def download_database(
    output_dir: Path,
    database: str,
    max_connections: int,
) -> None:
    script_path = files("genbio.aidosp.scripts").joinpath(
        DATABASE_TO_DOWNLOAD_SCRIPT[database]
    )
    with as_file(script_path) as script_path:
        result = subprocess.run(  # noqa: S603 since we can't get arbitrary input
            [str(script_path), output_dir, str(max_connections)],
            check=True,
        )
        if result.returncode != 0:
            raise click.ClickException(
                f"Failed to download {database} database. Please check the logs for more information."
            )
