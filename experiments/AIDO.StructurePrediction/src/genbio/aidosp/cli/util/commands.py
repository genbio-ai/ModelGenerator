import argparse
from pathlib import Path

import click
import genbio.aidosp.cli.util.download as dw_util
import ml_collections
import yaml
from genbio.aidosp.msa_retrieve.search_msa import main
from loguru import logger


@click.group()
def util() -> None:
    """Utility commands

    These commands are used to manage models, databases, and MSA retrieval.
    """
    pass


@util.command(help="Download the model checkpoint.")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="models/",
    help="Directory to save the downloaded model",
)
@click.option(
    "-mv",
    "--model-version",
    type=dw_util.ModelVersion,
    default="v0.1.2",
    help="Version of the model to download",
)
@click.option(
    "--repo-id",
    type=str,
    default="genbio-ai/AIDO.StructurePrediction",
    help="Repository ID of the model to download",
)
def download_model(output_dir: Path, model_version: str, repo_id: str) -> None:
    """Download the model checkpoint."""
    dw_util.download_model(output_dir, model_version, repo_id)


@util.command(help="Download the model checkpoint.")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="ccd/",
    help="Directory to save the downloaded ccd files",
)
@click.option(
    "--repo-id",
    type=str,
    default="genbio-ai/AIDO.StructurePrediction",
    help="Repository ID of the fiels to download",
)
def download_ccd(output_dir: Path, repo_id: str) -> None:
    """Download the model checkpoint."""
    dw_util.download_ccd(output_dir, repo_id)


@util.command()
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="db/",
    help="Directory to save the downloaded database files",
)
@click.option(
    "-n",
    "--name",
    type=dw_util.Database,
    multiple=True,
    default=["colabfold_envdb", "uniref30"],
    help="Name of the database to download",
)
@click.option(
    "-x",
    "--max-connections",
    type=click.IntRange(min=1, max=16),
    default=4,
    help="Maximum number of connections to use",
)
def download_db(output_dir: Path, name: list[str], max_connections: int) -> None:
    """Download the required database files."""
    logger.info(f"Downloading database files to {output_dir}")
    for n in name:
        logger.info(f"Downloading {n}")
        dw_util.download_database(output_dir, n, max_connections)


@util.command()
@click.option(
    "-i",
    "--input-fasta",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    required=True,
    help="Path to directory containing FASTA or one fasta file containing the sequences",
)
@click.option(
    "-c",
    "--config-yaml",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="Path to the config yaml file",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory to save the MSAs",
)
@click.option(
    "-j",
    "--cpus-per-task",
    type=click.IntRange(min=1),
    default=4,
    help="Number of CPUs per task",
)
@click.option(
    "-n",
    "--no-tasks",
    type=click.IntRange(min=1),
    default=1,
    help="Number of tasks to run",
)
@click.option(
    "--shuffle-file-list",
    is_flag=True,
    default=True,
    help="Whether to shuffle file list",
)
def retrieve_msa(
    input_fasta: Path,
    config_yaml: Path,
    output_dir: Path,
    cpus_per_task: int,
    no_tasks: int,
    shuffle_file_list: bool,
) -> None:
    """Retrieve MSA for the given protein sequences."""
    logger.info(f"Retrieving MSA for {input_fasta} and saving to {output_dir}")
    logger.info(f"Using config from {config_yaml}")

    with open(config_yaml) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    config = ml_collections.ConfigDict(hyp)
    args = argparse.Namespace()
    varargs = vars(args)
    # Needed for backward compatibility
    argdict = {
        "input": str(input_fasta),
        "output_dir": str(output_dir),
        "cpus_per_task": cpus_per_task,
        "raise_errors": False,
        "no_tasks": no_tasks,
        "shuffle_file_list": shuffle_file_list,
    }
    for k, v in argdict.items():
        varargs[k] = v
    main(args=args, config=config)
