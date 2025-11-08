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
