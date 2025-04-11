from Bio import PDB
import numpy as np
from Bio.SeqUtils import seq1
import sys
import pickle
import json

MAX_SEQ_LEN = 500 ## TODO: Hardcoded for now to match max-length of CATH 4.2. Change later.

def extract_backbone_coordinates(pdb_file, chain_id):
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
    except:
        parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Taking the first model
    
    backbone_atoms = ['N', 'CA', 'C', 'O']
    residues = [res for res in model[chain_id] if PDB.is_aa(res)]
    
    num_residues = min(MAX_SEQ_LEN, len(residues))
    coordinates = {atom: np.full((num_residues, 3), np.nan) for atom in backbone_atoms}
    residue_sequence = ""  # To store one-letter residue codes
    
    for i, residue in enumerate(residues):
        if i == MAX_SEQ_LEN: 
            break
        residue_sequence += seq1(residue.get_resname(), custom_map={})  # Convert three-letter to one-letter code
        for atom in backbone_atoms:
            if atom in residue:
                coordinates[atom][i] = residue[atom].coord
    
    num_chains = len(list(model.get_chains()))  # Count the number of chains
    
    return coordinates, residue_sequence, num_chains


if len(sys.argv) < 4:
    print("Usage: python preprocess_PDB.py <PDB_file_path> <PDB_CHAIN_ID>  <outdir>")
    sys.exit(1)

PDB_file_path = sys.argv[1] ## both .pdb or .cif file formats are acceptable
PDB_CHAIN_ID = sys.argv[2]
outdir = sys.argv[3]

PDB_ID = PDB_file_path.split('/')[-1].split('.')[0]

coordinates, residue_sequence, num_chains = extract_backbone_coordinates(pdb_file=PDB_file_path, chain_id=PDB_CHAIN_ID)
# print('Native sequence:')
# print(residue_sequence)
# print('coordinates')
# print(coordinates, coordinates['N'].shape, len(residue_sequence))

chain_set_map = {}
chain_set_map[PDB_ID+'.'+PDB_CHAIN_ID] = {}
chain_set_map[PDB_ID+'.'+PDB_CHAIN_ID]['seq'] = residue_sequence
chain_set_map[PDB_ID+'.'+PDB_CHAIN_ID]['coords'] = coordinates
chain_set_map[PDB_ID+'.'+PDB_CHAIN_ID]['num_chains'] = num_chains
chain_set_map[PDB_ID+'.'+PDB_CHAIN_ID]['name'] = PDB_ID+'.'+PDB_CHAIN_ID
chain_set_map[PDB_ID+'.'+PDB_CHAIN_ID]['CATH'] = None

chain_set_splits = {}
chain_set_splits['test'] = [PDB_ID+'.'+PDB_CHAIN_ID]
chain_set_splits['train'] = [PDB_ID+'.'+PDB_CHAIN_ID]
chain_set_splits['validation'] = [PDB_ID+'.'+PDB_CHAIN_ID]

pickle.dump(chain_set_map, open(outdir + 'chain_set_map.pkl', 'wb'))
json.dump(chain_set_splits, open(outdir + 'chain_set_splits.json', 'w'))
