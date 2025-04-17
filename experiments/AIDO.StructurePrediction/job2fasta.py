import os
import json

def json2fasta(input_json, output_fasta):
    with open(input_json, 'r') as f:
        data = json.load(f)

    fasta_lines = []
    for i, job in enumerate(data):
        for j, entity in enumerate(job['sequences']):
            if "proteinChain" in entity:
                sequence = entity['proteinChain']['sequence']
                fasta_lines.append(f">job_{i}_entity_{j}\n{sequence}")

    with open(output_fasta, 'w') as f:
        f.write("\n".join(fasta_lines))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Construct FASTA from job JSON for protein MSA retrieval.')
    parser.add_argument('--input', type=str, help='Input JSON file')
    parser.add_argument('--output', type=str, help='Output FASTA file')
    args = parser.parse_args()
    json2fasta(args.input, args.output)