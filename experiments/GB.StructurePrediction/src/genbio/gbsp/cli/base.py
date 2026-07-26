import warnings

import click
from genbio.gbsp.cli.completions.commands import completion
from genbio.gbsp.cli.util.commands import util


@click.group(help="GenBio GB Structure Prediction CLI")
def cli(): ...


cli.add_command(completion)
cli.add_command(util)


def cli_deprecated():
    """Deprecated entry point for the old ``genbio-aidosp`` console script.

    ``genbio-aidosp`` was renamed to ``genbio-gbsp`` in the AIDO.* -> GB.*
    rebrand. This alias emits a ``DeprecationWarning`` and then dispatches to
    the current CLI.
    """
    warnings.warn(
        "The 'genbio-aidosp' command is deprecated and will be removed in a "
        "future release; use 'genbio-gbsp' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cli()


if __name__ == "__main__":
    cli()
