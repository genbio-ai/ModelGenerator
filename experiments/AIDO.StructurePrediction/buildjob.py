import os
import glob
import json

def buildjob(input_json, input_msa_dir, output_json):
    # First, make a reverse index of sequences to MSA files
    msa_files = glob.glob(os.path.join(input_msa_dir, '*/*/*.fasta'))
    # key: protein sequence, values: precomputed_msa_dir, non_pairing_msa_names
    msa_dict = {}
    for msa_file in msa_files:
        # Get msa_file parent directory
        msa_dir = os.path.dirname(msa_file)
        msa_subdir = msa_dir.split(input_msa_dir)[-1].strip('/')
        # Get all the non-fasta files in the msa_dir
        msa_files = glob.glob(os.path.join(msa_dir, '*'))
        msa_files.remove(msa_file)
        msa_files = [os.path.basename(f) for f in msa_files]
        # Get the protein sequence from the fasta file
        with open(msa_file, 'r') as f:
            sequence = f.readlines()[1].strip()
        msa_dict[sequence] = {
            "precomputed_msa_dir": os.path.join('/msa_database', msa_subdir),
            "non_pairing_msa_names": msa_files
        }

    with open(input_json, 'r') as f:
        data = json.load(f)

    fasta_lines = []
    for i, job in enumerate(data):
        for j, entity in enumerate(job['sequences']):
            if "proteinChain" in entity:
                protein_chain = entity['proteinChain']
                sequence = protein_chain['sequence']
                # Add msa information
                assert sequence in msa_dict, f"Sequence {sequence} not found in MSA directory."
                protein_chain['msa'] = msa_dict[sequence]

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Construct FASTA from job JSON for protein MSA retrieval.')
    parser.add_argument('--input', type=str, help='Input job JSON file')
    parser.add_argument('--msa-db', type=str, help='Input directory for MSA files, built using job2msa.py')
    parser.add_argument('--output', type=str, help='Output job JSON file')
    args = parser.parse_args()
    buildjob(args.input, args.msa_db, args.output)