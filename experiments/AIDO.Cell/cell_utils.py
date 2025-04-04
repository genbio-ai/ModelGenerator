from typing import Tuple
import anndata as ad
import numpy as np
import os
import pandas as pd
import scanpy as sc


current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_GENES = pd.read_csv(os.path.join(current_dir, 'OS_scRNA_gene_index.19264.tsv'), sep='\t')['gene_name'].to_numpy()


def align_adata(adata: ad.AnnData) -> Tuple[ad.AnnData, np.ndarray]:
    """Aligns the input AnnData object to the AIDO.Cell gene set.

    Args:
        adata (ad.AnnData): The input AnnData object to be aligned.

    Returns:
        Tuple[ad.AnnData, np.ndarray]: A tuple containing the aligned AnnData object and an attention mask.
            The aligned AnnData object has the same genes as AIDO.Cell, and the attention mask indicates which genes
            are present in the AIDO.Cell pretraining set.
    """
    print('###########  Aligning data to AIDO.Cell  ###########')
    print(f'AIDO.Cell was pretrained on a fixed set of {len(MODEL_GENES)} genes.')
    print('Aligning your data to the AIDO.Cell gene set...')
    missing_genes = np.setdiff1d(MODEL_GENES, adata.var.index)
    new_missing_genes = np.setdiff1d(adata.var.index, MODEL_GENES)
    print(f'{len(new_missing_genes)} in your data that cannot be used by AIDO.Cell. Removing these.')
    print(new_missing_genes)
    print(f'{len(missing_genes)} genes in the AIDO.Cell pretraining set missing in your data.')
    print('AIDO.Cell is trained with zero-masking. Setting these to zero for AIDO.Cell to ignore.')
    print(missing_genes)
    adata_missing = ad.AnnData(np.zeros((adata.shape[0], len(missing_genes))))
    adata_missing.var.index = missing_genes
    adata_missing.obs = adata.obs
    adata_aligned = ad.concat((adata, adata_missing), axis=1, join='inner', merge='same')
    print(f'{len(MODEL_GENES) - len(missing_genes)} non-zero genes remaining.')
    print('Reordering genes to match AIDO.Cell gene ordering')
    adata_aligned = adata_aligned[:, MODEL_GENES]

    print('Gathering attention mask for nonzero genes')
    attention_mask = np.ones(adata_aligned.shape[1])
    attention_mask[np.isin(adata_aligned.var.index, missing_genes)] = 0
    print('####################  Finished  ####################')
    return adata_aligned, attention_mask

