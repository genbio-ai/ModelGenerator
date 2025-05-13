import anndata as ad
import bionty as bt
from lightning.pytorch.utilities import rank_zero_info
import numpy as np
import os
import pandas as pd

def build_map(gene_symbols):

    # Get map of symbols to Ensembl IDs:
    gene_map = bt.base.Gene(organism='human').standardize(
        gene_symbols,
        field='symbol',
        return_field='ensembl_gene_id',
        return_mapper=True,
    )

    # Identity map for any symbols that are already Ensembl IDs:
    for cur_symbol in gene_symbols:
        if cur_symbol.startswith('ENSG'):
            gene_map[cur_symbol] = cur_symbol
    
    return gene_map

def map_gene_symbols(adata: ad.AnnData, symbol_field='gene_symbol', ensembl_field='ensembl_id') -> ad.AnnData:
    '''
    Map gene symbols to Ensembl IDs using bionty.
    Args:
        adata: AnnData object.
        symbol_field: Column name in `adata.var` containing gene symbols.
        ensembl_field: Column name in `adata.var` to store Ensembl IDs.
    Returns:
        adata: AnnData object with Ensembl IDs added to `adata.var`.
    Notes:
        - Duplicate genes may be introduced.
        - If a gene symbol is already an Ensembl ID, it will be mapped to itself.
        - If a gene symbol cannot be mapped, it will be dropped from `adata.var`.
    '''

    if symbol_field == 'index':
        adata.var['index'] = adata.var.index

    gene_symbols = adata.var[symbol_field].to_numpy()

    gene_map = build_map(gene_symbols)
    
    # Drop columns for unmappable symbols:
    adata = adata[:, adata.var[symbol_field].isin(gene_map)]

    # Add ensembl IDs:
    adata.var[ensembl_field] = adata.var[symbol_field].map(gene_map)

    return adata

def align_genes(adata: ad.AnnData, ref_genes: np.ndarray, ensembl_field='ensembl_id') -> ad.AnnData:
    """Aligns the input AnnData object to a reference gene list.

    Args:
        adata (ad.AnnData): The input AnnData object to be aligned.
        ref_genes (np.ndarray): The reference gene list to align against.

    Returns:
        ad.AnnData: The aligned AnnData object.
    """
    rank_zero_info('###########  Aligning genes  ###########')
    rank_zero_info(f'Model was pretrained on a fixed set of {len(ref_genes)} genes.')
    adata.var[ensembl_field] = adata.var[ensembl_field].astype(str) # important because some gene symbols are <None> in some datasets
    non_dupe = ~adata.var[ensembl_field].duplicated()
    adata = adata[:, non_dupe]
    rank_zero_info(f'{np.sum(~non_dupe)} duplicate genes found in the data. Removing these based on occurrence order (this is arbitrary).')
    data_genes = adata.var[ensembl_field].to_numpy().astype(str)
    missing_genes = np.setdiff1d(ref_genes, data_genes)
    new_missing_genes = np.setdiff1d(data_genes, ref_genes)
    rank_zero_info(f'{len(new_missing_genes)} genes in the data that cannot be used by the model. Removing these.')
    rank_zero_info('e.g.', new_missing_genes[:10])
    rank_zero_info(f'{len(missing_genes)} model genes missing in the data. Zero padding.')
    rank_zero_info('e.g.', missing_genes[:10])
    adata_missing = ad.AnnData(np.zeros((adata.shape[0], len(missing_genes)), dtype=np.float32))
    adata_missing.var.index = missing_genes
    adata_missing.obs = adata.obs
    adata.var.index = data_genes
    adata_aligned = ad.concat((adata, adata_missing), axis=1, join='inner', merge='same')
    rank_zero_info(f'{len(ref_genes) - len(missing_genes)} non-zero genes remaining.')
    adata_aligned = adata_aligned[:, ref_genes]
    rank_zero_info('####################  Finished  ####################')
    return adata_aligned

def load_backbone_gene_list(backbone_class_path):
    '''Load the gene list used by the backbone, transforming to Ensembl IDs if necessary.
    
    Args:
        backbone_class_path: Path to the backbone class.
    
    Returns:
        genes: np.ndarray of ensembl IDs in the order used by the backbone. 
            Unmappable symbols are replaced with <symbol>_unknown_ensg; these will be zero-padded during gene alignment.
    '''
    load_base = os.path.dirname(os.path.abspath(__file__))
    model_name = backbone_class_path.split('.')[-1]
    if model_name in ['aido_cell_3m', 'aido_cell_10m', 'aido_cell_100m', 'scfoundation']:
        gene_symbols = pd.read_csv(os.path.join(load_base, 'gene_lists/OS_scRNA_gene_index.19264.tsv'), sep='\t')['gene_name'].values
        gene_map = build_map(gene_symbols)
        genes = np.array([gene_map.get(x, f'{x}_unknown_ensg') for x in gene_symbols])
    if model_name == 'geneformer':
        genes = pd.read_csv(os.path.join(load_base, 'gene_lists/geneformer_genes.csv'))['ensembl_id']
    else:
        raise NotImplementedError(f'Unknown gene set for backbone {backbone_class_path}.')
    return genes
