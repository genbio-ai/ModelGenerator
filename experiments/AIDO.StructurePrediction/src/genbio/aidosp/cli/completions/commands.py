from importlib.resources import files

import click


@click.command()
@click.option(
    "-s",
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default="bash",
    help="Shell to generate completion script for",
)
def completion(shell: str) -> None:
    """Generate shell completion script and exit

    This command generates a shell completion script for the GenBio AIDO Structure Prediction CLI.
    The generated script should be added to your shell configuration file:

    \b
    - bash: ~/.bashrc
    - zsh: ~/.zshrc
    - fish: ~/.config/fish/completions/genbio-aidosp.fish

    After modifying the shell config, you need to start a new shell in order for the changes to be loaded.
    """
    completions = files("genbio.aidosp.cli.completions")
    if shell == "bash":
        completion_script = completions.joinpath("genbio-aidosp-complete.bash")
    elif shell == "zsh":
        completion_script = completions.joinpath("genbio-aidosp-complete.zsh")
    elif shell == "fish":
        completion_script = completions.joinpath("genbio-aidosp.fish")
    else:
        raise click.BadParameter(f"Unsupported shell: {shell}")

    click.echo(completion_script.read_text())
