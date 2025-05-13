import os
import numpy as np
import pandas as pd
import anndata as ad
import tiledbsoma.io
from tqdm import tqdm

# Change the root path to your downloaded location
sctab_root_path = "merlin_cxg_2023_05_15_sf-log1p/"
# Output path
soma_exp_output_path = "soma-exp-scTab/"

if not os.path.isdir(soma_exp_output_path):
    os.makedirs(soma_exp_output_path)

for split in ["train", "val", "test"]:
    df = pd.DataFrame()
    for fname in tqdm(os.listdir(os.path.join(sctab_root_path, split)), desc=f"Loading {split} data files"):
        if not fname.endswith('.parquet'):
            continue
        fpath = os.path.join(sctab_root_path, split, fname)
        # Read the parquet file into a pandas DataFrame
        df = pd.concat([df, pd.read_parquet(fpath)])

    print("Converting ...")
    # Create AnnData object with the data
    adata = ad.AnnData(np.array(list(df['X'])))
    adata.obs = df[['cell_type']]
    adata.var = pd.read_parquet(os.path.join(sctab_root_path, "var.parquet"))
    # Save the data object into a TileDB experiment folder
    tiledbsoma.io.from_anndata(
        experiment_uri=os.path.join(soma_exp_output_path, split),
        measurement_name="RNA",
        anndata=adata
    )
    print(f"Data conversion for split '{split}' is done!"
)
