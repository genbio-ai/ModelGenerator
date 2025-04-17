
import ml_collections
import yaml
from genbio.aidosp.msa_retrieve.msar.utils.logger import Logger

logger = Logger.logger


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    config = ml_collections.ConfigDict(hyp)

    return config


def read_fasta(fasta_string: str):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions
