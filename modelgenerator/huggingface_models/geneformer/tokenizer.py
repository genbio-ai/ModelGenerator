"""
Geneformer tokenizer.
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from datasets import Dataset
from tqdm import tqdm
import torch

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")  # noqa

logger = logging.getLogger(__name__)

from . import ENSEMBL_MAPPING_FILE, GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE

def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]

def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # rank by median-scaled gene values
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])

class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1,
        chunk_size=512,
        model_input_size=4096,
        special_token=True,
        collapse_gene_ids=True,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        gene_mapping_file=ENSEMBL_MAPPING_FILE,
    ):
        """
        Initialize tokenizer.
        
        **Parameters:**
        
        custom_attr_name_dict : None, dict
            | Dictionary of custom attributes to be added to the dataset.
            | Keys are the names of the attributes in the loom file.
            | Values are the names of the attributes in the dataset.
        nproc : int
            | Number of processes to use for dataset mapping.
        chunk_size : int = 512
            | Chunk size for anndata tokenizer.
        model_input_size : int = 4096
            | Max input size of model to truncate input to.
            | For the 30M model series, should be 2048. For the 95M model series, should be 4096.
        special_token : bool = True
            | Adds CLS token before and EOS token after rank value encoding.
            | For the 30M model series, should be False. For the 95M model series, should be True.
        collapse_gene_ids : bool = True
            | Whether to collapse gene IDs based on gene mapping dictionary.
        gene_median_file : Path
            | Path to pickle file containing dictionary of non-zero median
            | gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            | Path to pickle file containing token dictionary (Ensembl IDs:token).
        gene_mapping_file : None, Path
            | Path to pickle file containing dictionary for collapsing gene IDs.

        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # chunk size for anndata tokenizer
        self.chunk_size = chunk_size

        # input size for tokenization
        self.model_input_size = model_input_size

        # add CLS and EOS tokens
        self.special_token = special_token

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # check for special token in gene_token_dict
        if self.special_token:
            if ("<cls>" not in self.gene_token_dict.keys()) and (
                "<eos>" not in self.gene_token_dict.keys()
            ):
                logger.error(
                    "<cls> and <eos> required in gene_token_dict when special_token = True."
                )
                raise

        if not self.special_token:
            if ("<cls>" in self.gene_token_dict.keys()) and (
                "<eos>" in self.gene_token_dict.keys()
            ):
                logger.warning(
                    "<cls> and <eos> are in gene_token_dict but special_token = False. Please note that for 95M model series, special_token should be True."
                )

        # if collapsing duplicate gene IDs
        self.collapse_gene_ids = collapse_gene_ids

        # load gene mappings dictionary (Ensembl IDs:Ensembl ID)
        if gene_mapping_file is not None:
            with open(gene_mapping_file, "rb") as f:
                self.gene_mapping_dict = pickle.load(f)
        else:
            self.gene_mapping_dict = {k: k for k, _ in self.gene_token_dict.items()}

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_token_dict.keys())

        #  Filter gene mapping dict for items that exist in gene_token_dict
        gene_keys_set = set(self.gene_token_dict.keys())
        self.gene_mapping_dict = {
            k: v for k, v in self.gene_mapping_dict.items() if v in gene_keys_set
        }

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_tensor(self, expression_data, gene_ids=None, target_sum=10_000):
        """
        Tokenize expression data directly from a tensor.
        
        **Parameters:**
        
        expression_data : torch.Tensor or numpy.ndarray
            | Expression data tensor with genes as features (columns)
        gene_ids : list, optional
            | List of Ensembl IDs corresponding to expression_data columns.
            | If None, assumes the expression data already contains only genes that need to be tokenized
        target_sum : int = 10_000
            | Target sum for normalization
            
        **Returns:**
        
        input_ids : list
            | List of tokenized input IDs for each cell
        """
        # Convert torch tensor to numpy if needed
        if isinstance(expression_data, torch.Tensor):
            expression_data = expression_data.cpu().numpy()
        
        # Get the number of genes in the expression data
        num_genes = expression_data.shape[1]
        

        if gene_ids is None:
           
            valid_gene_indices = list(range(min(num_genes, len(self.gene_keys))))
            valid_gene_ids = [self.gene_keys[i] for i in valid_gene_indices]
        else:
            # Filter to genes in our vocabulary
            valid_gene_indices = []
            valid_gene_ids = []
            for i, gene_id in enumerate(gene_ids):
                if gene_id in self.genelist_dict and self.genelist_dict[gene_id]:
                    valid_gene_indices.append(i)
                    valid_gene_ids.append(gene_id)
        
        if len(valid_gene_indices) == 0:
            raise ValueError("No valid genes found in input data")
            
        # Get normalization factors for selected genes
        norm_factor_vector = np.array([self.gene_median_dict.get(i, 1.0) for i in valid_gene_ids])
        
        # Get tokens for selected genes
        coding_miRNA_tokens = np.array([self.gene_token_dict[i] for i in valid_gene_ids])
        
        # Select only valid genes from expression data
        X_view = expression_data[:, valid_gene_indices]
        
        # Calculate total counts per cell (assume dense matrix for direct tensor input)
        n_counts = X_view.sum(axis=1, keepdims=True)
        n_counts = np.maximum(n_counts, 1.0)  # Avoid division by zero
        
        # Normalize by total counts and median expression
        X_norm = X_view / n_counts * target_sum / norm_factor_vector
        
        # Tokenize each cell
        input_ids = []
        for i in range(X_norm.shape[0]):
            # Get tokens for this cell
            cell_tokens = tokenize_cell(X_norm[i], coding_miRNA_tokens)
            
            # Apply truncation and special tokens if needed
            if self.special_token:
                cell_tokens = cell_tokens[0:self.model_input_size - 2]  # Truncate
                cell_tokens = np.insert(cell_tokens, 0, self.gene_token_dict.get("<cls>"))
                cell_tokens = np.append(cell_tokens, self.gene_token_dict.get("<eos>"))
            else:
                cell_tokens = cell_tokens[0:self.model_input_size]  # Truncate
                
            input_ids.append(cell_tokens)
            
        return input_ids

    def process_input_dict(self, input_data, gene_ids=None):
        """
        Process input dictionary containing expression data sequences.
        
        **Parameters:**
        
        input_dict : dict
            | Dictionary with "sequences" key containing expression data as torch.Tensor
            
        **Returns:**
        
        output_dict : dict
            | Dictionary with "input_ids" key containing tokenized input IDs
        """
        expression_data = input_data
        
        if expression_data is None:
            raise ValueError("Input dictionary must contain 'sequences' key")
            
        # Tokenize the expression data
        if gene_ids is None:
            input_ids = self.tokenize_tensor(expression_data, gene_ids=None)
        else:
            input_ids = self.tokenize_tensor(expression_data, gene_ids=gene_ids)
        
        # Return dictionary with input_ids
        return {"input_ids": input_ids}