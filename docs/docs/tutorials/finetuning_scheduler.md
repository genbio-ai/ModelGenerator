For some of our experiments, we leverage the [gradual unfreezing finetuning scheduler](https://github.com/genbio-ai/ModelGenerator/blob/main/modelgenerator/callbacks.py#L213), adapted from [RiNALMo](https://arxiv.org/abs/2403.00043)'s [code repository](https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/utils/finetune_callback.py).

#### Creating `schedule`
To use a FT scheduler, first we have to create a schedule and saving as a `.yaml` file. An example schedule is shown below:
```
0:
    - adapter.*
3:
    - backbone.encoder.encoder.ln.*
    - backbone.encoder.encoder.layer.32.*
```

In this example, when the model is setup, all the layers are first frozen. Then before the `0-th`-th epoch starts, all the parameters in the `adapter` module are unfrozen, and they remain unfrozen (trainable) for the rest of the training run. Similarly, before the `3-rd` epoch starts, parameters in the `backbone.encoder.encoder.ln` module (i.e., the last layer norm module of the backbone's encoder) is unfrozen, and they remain unfrozen until the training ends. Here can add any other layer or module if we want to unfreeze it before the starting of some specific epoch.

#### Using `schedule` when finetuning with ModelGenerator
In order to use this schedule for finetuning, we can simply to set this as CLI argument for `--trainer.callbacks.ft_schedule_path` when calling `mget fit`. 

Following is an example of finetuning the [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) model for RNA secondary structure prediction, with a **scheduler named `layers_0_32.yaml`**. (**NOTE:** Please refer to the [correspoding experiment folder](https://github.com/genbio-ai/ModelGenerator/tree/main/experiments/AIDO.RNA/rna_secondary_structure_prediction) for details of this experiment):
```
cd experiments/AIDO.RNA/rna_secondary_structure_prediction
MGEN_DATA_DIR=~/mgen_data
DATASET_NAME=bpRNA
CKPT_SAVE_DIR=logs/rna_ss/${DATASET_NAME}
mgen fit --config rna_ss_prediction.yaml \
    --data.path ${MGEN_DATA_DIR}/modelgenerator/datasets/rna_ss_data/ \
    --data.dataset ${DATASET_NAME} \
    --trainer.default_root_dir ${CKPT_SAVE_DIR} \
    --trainer.callbacks.ft_schedule_path ft_schedules/layers_0_32.yaml \
    --trainer.devices 0,1,2,3
```
