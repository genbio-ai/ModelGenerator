# Finetuning AIDO.Tissue for spatial single cell downstream tasks
In this file, we introduce how to finetune and evaluate our pre-trained CellSpatial foundation models for downstream tasks. These tasks can be classified into the following categories:

  * **Sequence-level classification tasks**: niche label type prediction
  * **Sequence-level regression tasks**: cell density prediction

Note: All the following scripts should be run under `ModelGenerator/`.

## Download data
The related data is deposited at https://huggingface.co/datasets/genbio-ai/tissue-downstream-tasks. Please download the data and put under `ModelGenerator/downloads` as `cell_density` or `niche_type_classification`. Under each sub-directory, there are three files denote different split (xx.train.h5ad, xx.val.h5ad, xx.test.h5ad).

For each `.h5ad`, several obs attributes should be included to reprezent the spatial (coordinate) information (like `x`, `y`), the label information (like `niche_label`). All the column fields will be specified in the following `config.yaml` file.

Note: the file `scRNA_genename_and_index.tsv` includes all the corresponding gene name and index in h5ad file.

## Sequence-level classification tasks
### niche label type prediction
We fully finetune AIDO.Tissue for niche label type prediction.


#### Finetuning script
```shell
CUDA_VISIBLE_DEVICES=7 nohup mgen fit --config experiments/AIDO.Tissue/niche_type_classfification.yaml > logs/nohup/AIDO.Tissue.niche_type_classfification.yaml.log 2>&1 &
```

Note:

The `filter_columns` includes label column and spatial coordinate column. `rename_columns` keep unchanged and will be used for running.


#### Evaluation script

Once finished run, there will be several `ckpt` file under the specified output directory `default_root_dir`. Then we can use the `ckpt` to evaluate on test dataset.

```shell
CUDA_VISIBLE_DEVICES=6 nohup mgen test --config experiments/AIDO.Tissue/niche_type_classfification.yaml \
  --ckpt_path ckpt_path \
  > ckpt_path.pred.log 2>&1 &
```

Note: `ckpt_path` is the finetuned checkpoint path.


## Sequence-level regression tasks

### cell density prediction

The config file is like `experiments/AIDO.Tissue/cell_density_regression.yaml`, all the fintuning running and evaluation are similar as classification task.

## Dump embedding

We can dump embedding for a `.h5ad` file. The script is as:

```shell
CUDA_VISIBLE_DEVICES=3 nohup mgen predict --config experiments/AIDO.Tissue/emb.xenium.yaml > logs/nohup/AIDO.Tissue.emb.xenium.log 2>&1 &
```

The output file will be under specified `output_dir` like `./logs/emb.xenium/lightning_logs/pred_output`. Each batch will be saved and a merged one will also be generated as `predict_predictions.pt`. The `predict_predictions.pt` file satcks all batches.

```shell
>>> import torch
>>> file_all = 'predict_predictions.pt'
>>> d_all = torch.load(file_all, map_location='cpu')
>>> d_all.keys()
dict_keys(['predictions', 'ids'])
>>> len(d_all['predictions']) # this equal to #sample
586
>>> len(d_all['ids']) # ids are numeric index corresponding to .h5ad file
586
>>> d_all['predictions'].shape # (B, L, D), L is max sequence length of all samples
torch.Size([586, 90, 128])
```

We can retrieve all the gene embedding and aggregate into cell embedding (like max pooling):

```bash
>>> d_all_maxpooling = [d_all['predictions'][i,:,:] for i in range(d_all['predictions'].shape[0])]
>>> d_all_maxpooling = [i[~torch.any(i.isnan(), dim=1)] for i in d_all_maxpooling]
>>> d_all_maxpooling = torch.cat([i.max(dim=0)[0].view(1,-1) for i in d_all_maxpooling])
>>> d_all_maxpooling.shape
torch.Size([586, 128])
```