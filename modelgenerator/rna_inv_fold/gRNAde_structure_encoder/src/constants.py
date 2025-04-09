import os

# import wandb

PROJECT_PATH = os.environ.get("PROJECT_PATH")

DATA_PATH = os.path.join(
    os.environ.get("MGEN_DATA_DIR"), "modelgenerator/datasets/rna_inv_fold/"
)
if DATA_PATH is None:
    raise Exception(
        "DATA_PATH not set properly. Did you set the environment variable 'MGEN_DATA_DIR'?"
    )

raw_data_dir = os.path.join(DATA_PATH, "raw_data")
if not os.path.exists(raw_data_dir):
    raise FileNotFoundError(
        f"The directory of raw data '{raw_data_dir}' does not exist."
    )

os.makedirs(os.path.join(DATA_PATH, "structure_encoding"), exist_ok=True)

MODEL_PATH = os.environ.get("gRNAde_CKPT_PATH")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"The gRNAde checkpoint file '{MODEL_PATH}' does not exist."
    )

SPITS_TO_CONSIDER = os.environ.get("SPITS_TO_CONSIDER")
if SPITS_TO_CONSIDER is None:
    SPITS_TO_CONSIDER = ["test"]
else:
    SPITS_TO_CONSIDER = SPITS_TO_CONSIDER.split(",")
    if SPITS_TO_CONSIDER[-1] == "":
        SPITS_TO_CONSIDER = SPITS_TO_CONSIDER[:-1]

X3DNA_PATH = os.environ.get("X3DNA")

ETERNAFOLD_PATH = os.environ.get("ETERNAFOLD")


# Value to fill missing coordinate entries when reading PDB files
FILL_VALUE = 1e-5


# Small epsilon value added to distances to avoid division by zero
DISTANCE_EPS = 0.001


# List of possible atoms in RNA nucleotides
RNA_ATOMS = [
    "P",
    "C5'",
    "O5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N1",
    "C2",
    "O2",
    "N2",
    "N3",
    "C4",
    "O4",
    "N4",
    "C5",
    "C6",
    "O6",
    "N6",
    "N7",
    "C8",
    "N9",
    "OP1",
    "OP2",
]


# List of possible RNA nucleotides
RNA_NUCLEOTIDES = [
    "A",
    "G",
    "C",
    "U",
    # '_'  # placeholder for missing/unknown nucleotides
]


# List of purine nucleotides
PURINES = ["A", "G"]


# List of pyrimidine nucleotides
PYRIMIDINES = ["C", "U"]


#
LETTER_TO_NUM = dict(zip(RNA_NUCLEOTIDES, list(range(len(RNA_NUCLEOTIDES)))))


#
NUM_TO_LETTER = {v: k for k, v in LETTER_TO_NUM.items()}


#
DOTBRACKET_TO_NUM = {".": 0, "(": 1, ")": 2}
