import click
from genbio.aidosp.cli.completions.commands import completion
from genbio.aidosp.cli.util.commands import util


@click.group(help="GenBio AIDO Structure Prediction CLI")
def cli(): ...


cli.add_command(completion)
cli.add_command(util)
if __name__ == "__main__":
    cli()
