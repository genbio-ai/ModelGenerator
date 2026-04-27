import pytest
import anndata as ad
import numpy as np
import pandas as pd
from modelgenerator.cell.utils import _ensure_unique_obs_names

def test_ensure_unique_obs_names():
    """Test that _ensure_unique_obs_names automatically resolves duplicate observation names."""
    # Arrange
    X = np.random.rand(3, 3)
    obs = pd.DataFrame(index=["cell_1", "cell_1", "cell_2"])
    
    # anndata 0.10.x requires explicit types for shape
    adata = ad.AnnData(X=X, obs=obs)

    # Pre-condition: names are not unique
    assert not adata.obs_names.is_unique

    # Act
    adata = _ensure_unique_obs_names(adata)

    # Assert: names should now be unique
    assert adata.obs_names.is_unique
