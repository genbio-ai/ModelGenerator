# Finetuning AIDO.RNA for RNA downstream tasks
In this file, we introduce how to finetune and evaluate our pre-trained RNA foundation models for various downstream tasks. These tasks can be classified into the following categories:
<!-- * Structure-related tasks:
    * RNA secondary structure prediction (sequence -> secondary structure)
    * RNA inverse folding (3D backbone -> sequence) -->

  * **Sequence-level regression tasks**: translation efficiency prediction, mRNA expression level prediction, transcript abundance prediction, protein abundance prediction
  * **Sequence-level classification tasks**: cross-species splice site prediction, ncRNA family classification, RNA modification site prediction

For details of task description, please refer to our paper: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1).

For access of our model and task datasets, please visit our Hugging Face collection [AIDO.RNA](https://huggingface.co/collections/genbio-ai/aidorna-6747516bb48ed96c847f5dd8).

Note: All the following scripts should be run under `ModelGenerator/`.

## Sequence-level regression tasks
### Translation efficiency prediction
We fully finetune AIDO.RNA-1.6B for mRNA translation efficiency predition using 10-fold cross validation on each of the cell line datasets, including 'Muscle', 'HEK', and 'pc3'. See `configs/translation_efficiency.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
CELL_LINE=Muscle

for FOLD in {0..9}
  do
    RUN_NAME=te_${CELL_LINE}_aido_rna_1b600m_fold${FOLD}
    CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}
    CUDA_VISIBLE_DEVICES=0 mgen fit --config experiments/AIDO.RNA/configs/translation_efficiency.yaml \
      --data.config_name translation_efficiency_${CELL_LINE} \
      --data.cv_test_fold_id $FOLD \
      --trainer.logger.name $RUN_NAME \
      --trainer.callbacks.dirpath $CKPT_SAVE_DIR
  done
```

Or simply launch finetuning using
```shell
bash experiments/AIDO.RNA/translation_efficiency_prediction.sh
```

Note:
1. The `data.cv_test_fold_id` defines which fold is used as test set.
2. The global batch size for 'Muscle', 'HEK', 'pc3' are 8, 32, 32, respectively.
3. It is run on 1 NVIDIA A100-80G GPU.

#### Evaluation script
```shell
CELL_LINE=Muscle

for FOLD in {0..9}
do
  CKPT_PATH=logs/rna_tasks/te_${CELL_LINE}_aido_rna_1b600m_fold${FOLD}/best_val*
  echo ">>> Fold ${FOLD}"
  mgen test --config experiments/AIDO.RNA/configs/translation_efficiency.yaml \
    --data.config_name translation_efficiency_${CELL_LINE} \
    --data.cv_test_fold_id $FOLD \
    --model.strict_loading True \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
done
```
Note: `ckpt_path` is the finetuned checkpoint path.


### mRNA expression level prediction
We fully finetune AIDO.RNA-1.6B for mRNA expression level prediction using 10-fold cross validation on each of the cell line datasets, including 'Muscle', 'HEK', and 'pc3'. See `configs/expression_level.yaml` for detailed hyperparameter settings. The finetuning and evaluation scripts are very similar to translation efficiency prediction.

Simply launch finetuning using
```shell
bash experiments/AIDO.RNA/expression_level_prediction.sh
```


### Transcript abundance prediction
We finetune AIDO.RNA-1.6B/AIDO.RNA-1.6B-CDS for transcript abundance prediction using LoRA and 5-fold cross validation on each of the organism datasets, including 'athaliana', 'dmelanogaster', 'ecoli', 'hsapiens', 'scerevisiae', 'ppastoris', 'hvolcanii'. See `configs/transcript_abundance.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
ORGANISM=ecoli
PROJECT=rna_tasks

for FOLD in {0..4}
do
  RUN_NAME=ta_${ORGANISM}_aido_rna_1b600m_fold${FOLD}
  CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
  mgen fit --config experiments/AIDO.RNA/configs/transcript_abundance.yaml \
    --data.config_name transcript_abundance_${ORGANISM} \
    --data.cv_test_fold_id $FOLD \
    --trainer.logger.name $RUN_NAME \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR
done
```

Or simply launch finetuning using
```shell
bash experiments/AIDO.RNA/transcript_abundance_prediction.sh
```

Note:
1. We can also use AIDO.RNA-1.6B-CDS as backbone for this task by setting `model.backbone` to `modelgenerator.backbones.aido_rna_1b600m_cds`.
2. The global batch sizes are all set to 16 for each tasks.
3. For LoRA finetuning, it will save lora weights and the prediction head weights only, making it light in disk space.
4. It is run on 4 NVIDIA A100-80G GPUs.

#### Evaluation script
```shell
ORGANISM=ecoli
PROJECT=rna_tasks

for FOLD in {0..4}
do
  RUN_NAME=ta_${ORGANISM}_aido_rna_1b600m_fold${FOLD}
  CKPT_PATH=logs/${PROJECT}/${RUN_NAME}/best_val*
  echo ">>> Fold ${FOLD}"
  mgen test --config experiments/AIDO.RNA/configs/transcript_abundance.yaml \
    --data.config_name transcript_abundance_${ORGANISM} \
    --data.cv_test_fold_id $FOLD \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
done
```
Note: For LoRA finetuning checkpoint, we need to load it with `model.strict_loading=False`.


### Protein abundance prediction
We finetune AIDO.RNA-1.6B/AIDO.RNA-1.6B-CDS for protein abundance prediction using LoRA and 5-fold cross validation on each of the organism datasets, including 'athaliana', 'dmelanogaster', 'ecoli', 'hsapiens', 'scerevisiae'. See `configs/protein_abundance.yaml` for detailed hyperparameter settings. We use global batch size 16 for each of the tasks. The finetuning and evaluation scripts are very similar to transcript abundance prediction .

Simply launch finetuning using
```shell
bash experiments/AIDO.RNA/protein_abundance_prediction.sh
```

# Mean Ribosome Load Prediction
Ribosomes are cellular structures responsible for protein synthesis, and the ribosome load on an mRNA molecule can influence the rate and efficiency of protein production, and the success of genetic engineering. Predicting ribosome load can provide valuable insights into gene expression regulation, translation efficiency, and cellular processes. We fully finetune AIDO.RNA-1.6B for mean ribosome load prediction using the dataset by [Sample _et al._](https://www.nature.com/articles/s41587-019-0164-5). We use the same train, test, and validation split used in a previous study [RiNALMo](https://arxiv.org/abs/2403.00043). See the [config file](https://github.com/genbio-ai/ModelGenerator/tree/main/experiments/AIDO.RNA/mean_ribosome_load_prediction/mean_ribosome_load_prediction.yaml) for detailed hyperparameter settings.

#### Finetuning script
```shell
RUN_NAME=rna_mrl
CKPT_SAVE_DIR=logs/${RUN_NAME}
mgen fit --config experiments/AIDO.RNA/mean_ribosome_load_prediction/mean_ribosome_load_prediction.yaml \
           --trainer.default_root_dir ${CKPT_SAVE_DIR} \
           --trainer.callbacks.ft_schedule_path experiments/AIDO.RNA/mean_ribosome_load_prediction/ft_schedules/two_step.yaml \
           --trainer.devices 0,
```

Note that here we are using finetuning scheduler. See [this tutorial](https://github.com/genbio-ai/ModelGenerator/blob/main/docs/docs/tutorials/finetuning_scheduler.md) for details.

#### Evaluation script
```shell
RUN_NAME=rna_mrl
CKPT_SAVE_DIR=logs/${RUN_NAME}
CKPT_PATH=/path/to/checkpoint ## NOTE: Replace `/path/to/checkpoint` with the actual finetuned checkpoint path.
mgen test --config experiments/AIDO.RNA/mean_ribosome_load_prediction/mean_ribosome_load_prediction.yaml \
           --trainer.default_root_dir ${CKPT_SAVE_DIR}/test \
           --trainer.callbacks.ft_schedule_path experiments/AIDO.RNA/mean_ribosome_load_prediction/ft_schedules/two_step.yaml \
           --trainer.devices 0, \
           --ckpt_path ${CKPT_PATH}
```

## Sequence-level classification tasks

### Cross-species splice site prediction
We finetune AIDO.RNA-1.6B for splice site prediction on 'accecptor' and 'donor' datasets using LoRA. See `configs/splice_site_prediction.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
SPLICE_SITE=acceptor
RUN_NAME=csp_${SPLICE_SITE}_aido_rna_1b600m
CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}

CUDA_VISIBLE_DEVICES=0,1 mgen fit --config experiments/AIDO.RNA/configs/splice_site_prediction.yaml \
  --data.config_name splice_site_${SPLICE_SITE} \
  --trainer.logger.name $RUN_NAME \
  --trainer.callbacks.dirpath $CKPT_SAVE_DIR
```

Or simply launch finetuning using
```shell
bash experiments/AIDO.RNA/splice_site_prediction.sh
```

Note:
1. The global batch sizes are set to 32 for both tasks.
2. It is run on 2 NVIDIA A100-80G GPUs.

#### Evaluation script
```shell
SPLICE_SITE=acceptor
RUN_NAME=csp_${SPLICE_SITE}_aido_rna_1b600m
CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}
CKPT_PATH=logs/rna_tasks/${RUN_NAME}/best_val*

for TEST_TYPE in danio fly worm thaliana
do
  echo $TEST_TYPE
  mgen test --config experiments/AIDO.RNA/configs/splice_site_prediction.yaml \
    --data.config_name splice_site_${SPLICE_SITE} \
    --data.test_split_name test_$TEST_TYPE \
    --data.batch_size 256 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATHs
done
```

### ncRNA family classification
We finetune AIDO.RNA-1.6B for ncRNA family classification on 'bnoise0' and 'bnoise200' datasets using LoRA. See `configs/ncrna_family_classification.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
BOUNDARY_NOISE=bnoise0
RUN_NAME=nfc_${BOUNDARY_NOISE}_aido_rna_1b600m
CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}

CUDA_VISIBLE_DEVICES=0,1 mgen fit --config experiments/AIDO.RNA/configs/ncrna_family_classification.yaml \
  --data.config_name ncrna_family_${BOUNDARY_NOISE} \
  --trainer.callbacks.dirpath $CKPT_SAVE_DIR
```

Or simply launch finetuning using
```shell
bash experiments/AIDO.RNA/ncrna_family_classification.sh
```

Note:
1. The global batch sizes are set to 64 for both tasks.
2. It is run on 2 NVIDIA A100-80G GPUs.

#### Evaluation script
```shell
SPLICE_SITE=acceptor
RUN_NAME=csp_${SPLICE_SITE}_aido_rna_1b600m
CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}
CKPT_PATH=logs/rna_tasks/${RUN_NAME}/best_val*

for TEST_TYPE in danio fly worm thaliana
do
  echo $TEST_TYPE
  mgen test --config experiments/AIDO.RNA/configs/splice_site_prediction.yaml \
    --data.config_name splice_site_${SPLICE_SITE} \
    --data.test_split_name test_$TEST_TYPE \
    --data.batch_size 256 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATHs
done
```


### RNA modification site prediction
We finetune AIDO.RNA-1.6B for RNA modification site prediction using LoRA. See `configs/ncrna_family_classification.yaml` for detailed hyperparameter settings.

#### Finetuning script
```shell
RUN_NAME=msp_aido_rna_1b600m
PROJECT=rna_tasks
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CONFIG_FILE=experiments/AIDO.RNA/configs/modification_site_prediction.yaml

srun mgen fit --config $CONFIG_FILE \
  --trainer.logger.name $RUN_NAME \
  --trainer.logger.project $PROJECT \
  --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
  --trainer.num_nodes 4
```
Note:
1. The global batch size is set to 64.
2. It is run on 4*4 NVIDIA A100-80G GPUs using slurm system.

#### Evaluation script
```shell
RUN_NAME=msp_aido_rna_1b600m
PROJECT=rna_tasks
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CONFIG_FILE=experiments/AIDO.RNA/configs/modification_site_prediction.yaml
CKPT_PATH=${CKPT_SAVE_DIR}/best_val*

mgen test --config $CONFIG_FILE \
  --data.batch_size 16 \
  --model.strict_loading False \
  --model.reset_optimizer_states True \
  --trainer.logger null \
  --ckpt_path $CKPT_PATH
```
