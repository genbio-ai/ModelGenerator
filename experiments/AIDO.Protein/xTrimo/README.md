# Finetuning AIDO.Protein for protein downstream tasks

In this file, we introduce how to finetune and evaluate our pre-trained protein language models for various downstream tasks from xTrimoPGLM benchmark. For details of task description, please refer to our paper: [Mixture of Experts Enable Efficient and Effective Protein Understanding and Design](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1).

For access of our model and task datasets, please visit our Hugging Face collection [AIDO.Protein](https://huggingface.co/collections/genbio-ai/aidoprotein-6747522bc86c9ee23472b703).

Note: All the following scripts should be run under `ModelGenerator/`.

## Contact prediction
We finetune AIDO.Protein-16B for contact prediction using LoRA. See `contact_prediction_binary.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
RUN_NAME=AIDO.Protein_16B_fsdp_bs4
PROJECT=xtrimo_benchmark
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/contact_prediction_binary.yaml
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}

srun mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --trainer.num_nodes 1
```
Note:
1. We use FSDP with a global batch size of 4.
2. It is run on 4 NVIDIA A100-80G GPU using slurm system.

#### Evaluation script
```shell
RUN_NAME=AIDO.Protein_16B_fsdp_bs4
PROJECT=xtrimo_benchmark
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/contact_prediction_binary.yaml
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CKPT_PATH=${CKPT_SAVE_DIR}/best_val*.ckpt

mgen test --config $CONFIG_FILE  \
    --data.batch_size 1 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
```
Note: `ckpt_path` is the finetuned checkpoint path.


## Secondary structure prediction
We finetune AIDO.Protein-16B for secondary structure prediction using LoRA. See `ssp_q3.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
RUN_NAME=ssp_AIDO.Protein_16B
PROJECT=xtrimo_benchmark
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/ssp_q3.yaml
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}

srun mgen fit --config $CONFIG_FILE \
--trainer.logger.name $RUN_NAME \
--trainer.logger.project $PROJECT \
--trainer.callbacks.dirpath $CKPT_SAVE_DIR \
--trainer.num_nodes 2
```
Note: It is run on 2*4 NVIDIA A100-80G GPU using slurm system.

#### Evaluation script
```shell
RUN_NAME=ssp_AIDO.Protein_16B
PROJECT=xtrimo_benchmark
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/ssp_q3.yaml
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CKPT_PATH=${CKPT_SAVE_DIR}/best_val*

mgen test --config $CONFIG_FILE  \
    --data.batch_size 4 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
```


## Sequence-level regression

### Fold prediction
We finetune AIDO.Protein-16B for fold prediction using LoRA. See `fold_prediction.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
RUN_NAME=fold_AIDO.Protein_16B
PROJECT=xtrimo_benchmark
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/fold_prediction.yaml
srun mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --trainer.num_nodes 4
```
Note: It is run on 4*4 NVIDIA A100-80G GPU using slurm system.

#### Evaluation script
```shell
RUN_NAME=fold_AIDO.Protein_16B
PROJECT=xtrimo_benchmark
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/fold_prediction.yaml
CKPT_PATH=${CKPT_SAVE_DIR}/best_val*

mgen test --config $CONFIG_FILE \
    --data.batch_size 16 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
```


## Sequence-level classification

### Fluorescence prediction
We finetune AIDO.Protein-16B for fluorescence prediction using LoRA. See `fluorescence_prediction.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
RUN_NAME=fluorescence_AIDO.Protein_16B
PROJECT=xtrimo_benchmark
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/fluorescence_prediction.yaml
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}

srun mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --trainer.num_nodes 2
```
Note: It is run on 2*4 NVIDIA A100-80G GPU using slurm system.

#### Evaluation script
```shell
RUN_NAME=fluorescence_AIDO.Protein_16B
PROJECT=xtrimo_benchmark
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/fluorescence_prediction.yaml
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CKPT_PATH=${CKPT_SAVE_DIR}/best_val*

mgen test --config $CONFIG_FILE \
    --data.batch_size 16 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
```
