# Takes a directory with *.pt files as argument
# For each file, loads the embeddings and computes the mean along the sequence dimension
# Compiles all mean embeddings as well as sequences into a single pt file

import sys
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm

def compile_mean_embeddings(directory):
    all_sequences = []
    mean_embeddings = []
    for file in tqdm(os.listdir(directory)):
        if file.endswith('.pt'):
            vals = torch.load(os.path.join(directory, file))
            embeddings = torch.tensor(vals['predictions']).cpu()
            sequences = vals['sequences']
            attention_masks = torch.tensor(vals['attention_mask']).cpu()
            special_tokens_mask = torch.tensor(vals['special_tokens_mask']).cpu()
            for i in range(len(embeddings)):
                embedding = embeddings[i][attention_masks[i] == 1 & ~special_tokens_mask[i]]
                mean_embedding = embedding.mean(dim=0)
                mean_embeddings.append(mean_embedding)
                all_sequences.append(sequences[i])
    mean_embeddings = torch.stack(mean_embeddings)
    torch.save({'sequences': all_sequences, 'mean_embeddings': mean_embeddings}, os.path.join(directory, 'mean_embeddings.pt'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile mean embeddings')
    parser.add_argument('--directory', type=str, help='Path to directory with *.pt files')
    args = parser.parse_args()
    compile_mean_embeddings(args.directory)