import anndata as ad
import numpy as np
import torch
import sys
from modelgenerator.tasks import Embed

device = 'cuda'
batch_size = 4

model = Embed.from_config({
        "model.backbone": "aido_cell_3m",
        "model.batch_size": batch_size
    }).eval()
model = model.to(device).to(torch.float16)

adata = ad.read_h5ad('../../modelgenerator/cell-downstream-tasks/zheng/zheng_train.h5ad')

batch_np = adata[:batch_size].X.toarray()
batch_tensor = torch.from_numpy(batch_np).to(torch.float16).to(device)
batch_transformed = model.transform({'sequences': batch_tensor})
embs = model(batch_transformed)

print(embs)
