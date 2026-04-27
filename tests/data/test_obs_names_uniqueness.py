import anndata as ad
import numpy as np
import pytest
import logging
from modelgenerator.cell.utils import _ensure_unique_obs_names

def test_make_obs_names_unique():
    # Create AnnData with duplicate obs_names
    data = np.random.rand(3, 2)
    obs_names = ["cell1", "cell1", "cell2"]
    adata = ad.AnnData(data)
    adata.obs_names = obs_names

    assert not adata.obs_names.is_unique

    # Apply fix
    _ensure_unique_obs_names(adata)

    assert adata.obs_names.is_unique
    assert list(adata.obs_names) == ["cell1", "cell1-1", "cell2"]
    
def test_ensure_unique_obs_names_no_duplicates():
    # Create AnnData with unique obs_names
    data = np.random.rand(2, 2)
    obs_names = ["cell1", "cell2"]
    adata = ad.AnnData(data)
    adata.obs_names = obs_names

    assert adata.obs_names.is_unique

    # Apply fix
    _ensure_unique_obs_names(adata)

    assert adata.obs_names.is_unique
    assert list(adata.obs_names) == ["cell1", "cell2"]

if __name__ == "__main__":
    test_make_obs_names_unique()
    test_ensure_unique_obs_names_no_duplicates()
    print("All tests passed!")
