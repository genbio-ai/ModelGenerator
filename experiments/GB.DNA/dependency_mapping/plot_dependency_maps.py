"""
Script for plotting outputs from dependency mapping
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logomaker


def plot_depmap(dependency_matrix, influence_scores, seq_wt, vocab, savepath):
    # First make the heatmap, then the logo
    plt.rcParams.update({"font.size": 18})
    plt.figure(figsize=(24, 20))
    sns.heatmap(dependency_matrix, cmap="coolwarm", cbar_kws={"label": "Log Odds Ratio"})
    plt.xticks(np.arange(len(seq_wt)) + 0.5, seq_wt, fontsize=16 * 72 / len(seq_wt))
    plt.yticks([])
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath + "_heatmap.png", dpi=300)
        plt.savefig(savepath + "_heatmap.pdf")
    plt.close()

    # Saliency logo, where letter size is the maximum off-diagonal log-odds change
    # Make saliency matrix
    nn_df = pd.DataFrame(influence_scores, columns=vocab)
    # Make the logo plot
    plt.rcParams.update({"font.size": 12})
    nn_logo = logomaker.Logo(nn_df, figsize=(len(seq_wt) // 5, 2))
    nn_logo.style_spines(visible=False)
    nn_logo.ax.set_xticks([])
    nn_logo.ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(savepath + "_logo.png", dpi=300)


def main(input_dir, output_dir, vocab, tokens):
    # Keys to keep
    keys = set(["ids", "input_ids", "predictions", "pos_i", "mut_i", "sequences"])
    # Get token ids in order of mutation vocab
    tokens_to_keep = torch.Tensor([tokens.index(v) for v in vocab]).long()
    # Transform batches of dicts of lists into a single list of sample dicts
    ids = set()
    data = []
    for file in os.listdir(input_dir):
        if not file.endswith(".pt"):
            continue
        batch_dict = torch.load(os.path.join(input_dir, file))
        batch_list = [{k: v[i] for k, v in batch_dict.items() if k in keys} for i in range(len(batch_dict["ids"]))]
        data.extend(batch_list)
        ids.update(batch_dict["ids"])

    # Filter into a dict of lists of sample dicts for each id
    def clean_sample(sample):
        # Keep only the inputs and channels in the mutation vocabulary
        input_ids = sample["input_ids"].cpu()
        predictions = sample["predictions"].cpu()
        keep_idx = torch.Tensor([i in tokens_to_keep for i in input_ids]).bool()
        sample["input_ids"] = input_ids[keep_idx]
        sample["predictions"] = predictions[keep_idx][:, tokens_to_keep]
        return sample

    data_by_id = {id: [] for id in ids}
    wt_by_id = {}
    for sample in tqdm(data):
        sample = clean_sample(sample)
        id = sample["ids"]
        if sample["mut_i"] == -1:
            wt_by_id[id] = sample
        else:
            data_by_id[id].append(sample)

    # For each id, get the dependency matrices and plot the dependency maps
    def make_dependency_mat(samples, wt, vocab, tokens, eps=1e-4):
        # Get the dependency matrices
        seq_len = len(wt["input_ids"])
        # Potential for error here, since we discarded logits before softmax instead of normalizing after
        probs_wt = F.softmax(wt["predictions"], dim=-1)
        probs_wt = probs_wt + eps
        probs_wt = probs_wt / torch.sum(probs_wt, dim=-1, keepdim=True)
        probs_wt = probs_wt.detach().cpu().numpy()
        odds_wt = probs_wt / (1 - probs_wt)
        probs_muts = np.zeros((seq_len, len(vocab), seq_len, len(vocab)))  # L x V x L x V
        for sample in samples:
            probs_mut = F.softmax(sample["predictions"], dim=-1)
            probs_mut = probs_mut + eps
            probs_mut = probs_mut / torch.sum(probs_mut, dim=-1, keepdim=True)
            probs_mut = probs_mut.detach().cpu().numpy()
            probs_muts[sample["pos_i"], sample["mut_i"]] = probs_mut
        odds_muts = probs_muts / (1 - probs_muts)
        dependency_matrix_full = np.abs(np.log2(odds_muts / odds_wt))
        for i in range(dependency_matrix_full.shape[0]):  # Mask self-mutation dependencies
            dependency_matrix_full[i, :, i, :] = 0
        # Get the maximum log odds ratio for each nucleotide over all possible mutations and all nucleotide probabilities
        dependency_matrix = dependency_matrix_full.max(axis=(1, 3))
        # Compute the "score" for each nuc at each position by getting the entropy-weighted probability matrix.
        background = np.array([0.25, 0.25, 0.25, 0.25])
        background_ppl = np.log2(background) * background
        ic = np.log2(probs_wt) * probs_wt - background_ppl[np.newaxis, :]
        ic = ic.sum(axis=1)
        nuc_weights = probs_wt * ic[:, np.newaxis]
        # Plot the heatmap and the score logo!
        return dependency_matrix, nuc_weights

    for id in ids:
        samples = data_by_id[id]
        wt = wt_by_id[id]
        dependency_matrix, nuc_weights = make_dependency_mat(samples, wt, vocab, tokens)
        plot_depmap(
            dependency_matrix, nuc_weights, wt["sequences"], vocab, os.path.join(output_dir, str(wt["ids"]))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot dependency maps")
    parser.add_argument("-i", help="Path to the input files directory")
    parser.add_argument("-o", help="Path to the output files directory")
    parser.add_argument("-v", help="Path to the file with the mutation vocabulary")
    parser.add_argument("-t", help="Path to the file with the tokenizer vocabulary")
    args = parser.parse_args()
    vocab = open(args.v).read().strip().split("\n")
    tokens = open(args.t).read().strip().split("\n")
    os.makedirs(args.o, exist_ok=True)
    main(args.i, args.o, vocab, tokens)
