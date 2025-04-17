
from pathlib import Path

COMPONENTS_FILE = None
RKDIT_MOL_PKL = None

def set_components_file(components_file):
    global COMPONENTS_FILE
    COMPONENTS_FILE = components_file

def set_rkdit_mol_pkl(rkdit_mol_pkl):
    global RKDIT_MOL_PKL
    RKDIT_MOL_PKL = Path(rkdit_mol_pkl)


def get_components_file():
    global COMPONENTS_FILE
    return COMPONENTS_FILE

def get_rkdit_mol_pkl():
    global RKDIT_MOL_PKL
    return RKDIT_MOL_PKL
