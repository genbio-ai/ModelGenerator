# A Modular Framework for Multi-Modal Learning with Foundation Models
 
This task combines DNA and RNA foundation models to predict the expression of an RNA isoform across 30 tissues. 
The original task was proposed by [Garau-Luis et al.](https://arxiv.org/html/2406.14150v1)
We implemented this task in ModelGenerator as the `MMSequenceRegression` task, which allows up to 3 foundation models to be combined toward a supervised objective.

The task enables concatenation or cross-attention on the embeddings, plus a final output head which can be swapped or modified modularly.
The code is implemented in a model- and task-agnostic way, allowing flexible combinations of FMs across a wide range of downstream applications.

## Baselines: Single-modality Regression

### RNA-only model with AIDO.RNA-1.6B
```
mgen fit --model SequenceRegression \
    --model.backbone aido_rna_1b600m \
    --model.adapter MLPPoolAdapter \
    --model.num_outputs 30 \
    --model.use_peft True \
    --data IsoformExpression

mgen test --model SequenceRegression \
    --model.backbone aido_rna_1b600m \
    --model.adapter MLPPoolAdapter \
    --model.num_outputs 30 \
    --data IsoformExpression \
    --model.strict_loading False \
    --ckpt_path <your-ckpt-path>.ckpt
  
# Spearman: 0.56
```

### DNA-only model with Enformer
```
mgen fit --model SequenceRegression \
    --model.backbone enformer \
    --model.adapter MLPPoolAdapter \
    --model.num_outputs 30 \
    --model.use_peft True \
    --data IsoformExpression

mgen test --model SequenceRegression \
    --model.backbone enformer \
    --model.adapter MLPPoolAdapter \
    --model.num_outputs 30 \
    --data IsoformExpression \
    --model.strict_loading False \
    --ckpt_path <your-ckpt-path>.ckpt

# Spearman 0.66
```

### Multimodal DNA + DNA model with AIDO.RNA and Enformer
This model captures regulatory context and relates it to the RNA isoform of interest.
```
mgen fit --config configs/isoform_expression_dna_rna_concat.yaml
# E.g.
# mgen fit --model MMSequenceRegression \
#     --model.backbone enformer \
#     --model.backbone1 aido_rna_1b600m \
#     --model.adapter.fusion ConcatFusion \
#     --model.adapter.adapter MLPAdapter \
#     --model.num_outputs 30
#     --model.use_peft true \
#     --data IsoformExpression

mgen test --config configs/isoform_expression_dna_rna_concat.yaml \
    --model.strict_loading false \
    --ckpt_path <your-ckpt-path>.ckpt

# Spearman 0.73
```

## Key features of `MMSequenceRegression`
1. **Flexible backbone combination**

    Supports fusion of 2–3 backbones, for example:
    * (Enformer, AIDO.RNA)
    * (Enformer, AIDO.RNA, ESM2)

    Note: The backbone does not need to be an AIDO model. Any compatible model supported by `ModelGenerator` can be used.

2. **Supported fusion methods**
    * `CrossAttentionFusion`
    * `ConcatFusion`

3. **Supported tasks**
    * Multi-modal RNA isoform expression prediction [1]
    * Easily extensible to other tasks such as:
        * Protein–protein interaction prediction
        * Protein–RNA/DNA interaction prediction


## Usage example: multi-modal RNA isoform expression prediction
This is a multi-modal, multi-label regression task. The input consists of any combination of:
* DNA sequence
* RNA sequence
* Protein sequence

The target is the RNA expression level across 30 human tissues. 

### Fusion of DNA and RNA FMs
We take DNA and RNA sequences as input. We use Enformer for DNA sequences and AIDO.RNA for RNA sequences. The fusion method here is cross-attention fusion. We fully finetune Enformer and AIDO.RNA during the training.

**Configs:**
```
model:
  class_path: modelgenerator.tasks.MMSequenceRegression
  init_args:
    backbone:
      class_path: modelgenerator.backbones.enformer
      init_args:
        max_length: 196_608
        frozen: false
    backbone1:
      class_path: modelgenerator.backbones.aido_rna_650m
      init_args:
        from_scratch: false
        max_length: 1024
        frozen: false
        use_peft: false
        save_peft_only: false
        config_overwrites:
          hidden_dropout_prob: 0.1
          attention_probs_dropout_prob: 0.1
        model_init_args: null
    backbone_order:
    - dna_seq
    - rna_seq
    adapter:
      class_path: modelgenerator.adapters.fusion.MMFusionSeqAdapter
      init_args:
        fusion:
          class_path: modelgenerator.adapters.fusion.CrossAttentionFusion
          init_args:
            num_attention_heads: 16
        adapter:
          class_path: modelgenerator.adapters.MLPPoolAdapter
          init_args:
            pooling: mean_pooling
            hidden_sizes:
            - 128
            bias: true
            dropout: 0.1
            dropout_in_middle: false
    num_outputs: 30
```
For full configs, see `ModelGenerator/experiments/AIDO.RNA/multimodal_isoform_expression/configs/isoform_expression_dna_rna_attention.yaml`.

**Training script:**
```
RUN_NAME=enformer_aidorna650m
CONFIG_FILE=experiments/AIDO.RNA/multimodal_isoform_expression/configs/isoform_expression_dna_rna_attention.yaml
PROJECT=isoform_tasks
CKPT_SAVE_DIR=${GENBIO_DATA_DIR}/genbio_finetune/logs/${PROJECT}/${RUN_NAME}

mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --model.optimizer.lr 1e-4 \
    --data.batch_size 2 \
```

**Evaluation script:**
```
CONFIG_FILE=experiments/AIDO.RNA/multimodal_isoform_expression/configs/isoform_expression_dna_rna_attention.yaml
CKPT_PATH=...        # input the checkpoint path here

mgen test --config $CONFIG_FILE \
    --data.batch_size 16 \
    --trainer.logger null \
    --model.strict_loading True \
    --model.reset_optimizer_states True \
    --ckpt_path $CKPT_PATH
```


### Fusion of DNA, RNA and protein FMs
We take DNA, RNA and protein sequences as input. We use Enformer for DNA sequences, AIDO.RNA for RNA sequences, and ESM2 for protein sequences. The fusion method here is concat fusion.  We fully finetune Enformer while using LoRA finetuning for AIDO.RNA and ESM2 during the training.

**Configs:**
```
model:
  class_path: modelgenerator.tasks.MMSequenceRegression
  init_args:
    backbone:
      class_path: modelgenerator.backbones.enformer
      init_args:
        max_length: 196_608
        frozen: false
    backbone1:
      class_path: modelgenerator.backbones.aido_rna_1b600m_cds
      init_args:
        max_length: 1024
        frozen: false 
        use_peft: true  
        save_peft_only: true 
        lora_r: 32
        lora_alpha: 64
        lora_dropout: 0.1
        lora_target_modules:
        - query
        - value
    backbone2:
      class_path: modelgenerator.backbones.esm2_150m
      init_args:
        max_length: 1024
        frozen: false
        use_peft: true  
        save_peft_only: true 
        lora_r: 32
        lora_alpha: 64
        lora_dropout: 0.1
        lora_target_modules:
        - query
        - value
    backbone_order:
    - dna_seq
    - rna_seq
    - protein_seq
    adapter:
      class_path: modelgenerator.adapters.fusion.MMFusionTokenAdapter
      init_args:
        fusion:
          class_path: modelgenerator.adapters.fusion.ConcatFusion
          init_args:
            project_size: 1024
            pooling: mean_pooling
        adapter:
          class_path: modelgenerator.adapters.MLPAdapter
          init_args:
            hidden_sizes:
            - 1024
            bias: true
            dropout: 0.1
            dropout_in_middle: false
    num_outputs: 30
```
For full configs, see `ModelGenerator/experiments/AIDO.RNA/multimodal_isoform_expression/configs/isoform_expression_dna_rna_prot_concat.yaml`.


**Training script:**
```
RUN_NAME=enformer_aidorna1.6b_esm2
CONFIG_FILE=experiments/AIDO.RNA/multimodal_isoform_expression/configs/isoform_expression_dna_rna_prot_concat.yaml
PROJECT=isoform_tasks
CKPT_SAVE_DIR=${GENBIO_DATA_DIR}/genbio_finetune/logs/${PROJECT}/${RUN_NAME}

mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --model.optimizer.lr 1e-4 \
    --data.batch_size 1
```

**Evaluation script:**
```
CONFIG_FILE=experiments/AIDO.RNA/multimodal_isoform_expression/configs/isoform_expression_dna_rna_prot_concat.yaml
CKPT_PATH=...

mgen test --config $CONFIG_FILE \
    --data.batch_size 16 \
    --trainer.logger null \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --ckpt_path $CKPT_PATH
```


### Usage of released checkpoints on Hugging Face
We released two state-of-the-art checkpoints for multi-modal RNA isoform expression prediction, which are part of the results in our AIDO.RNA manuscript. The checkpoints are as follows:
* [genbio-ai/IsoformExpression-Enformer-AIDO.RNA-1.6B-CDS-ESM2-150M-ConcatFusion-rna-isoform-expression-ckpt](https://huggingface.co/genbio-ai/IsoformExpression-Enformer-AIDO.RNA-1.6B-CDS-ESM2-150M-ConcatFusion-rna-isoform-expression-ckpt)
* [genbio-ai/IsoformExpression-Enformer-AIDO.RNA-1.6B-CDS-ConcatFusion-rna-isoform-expression-ckpt](https://huggingface.co/genbio-ai/IsoformExpression-Enformer-AIDO.RNA-1.6B-CDS-ConcatFusion-rna-isoform-expression-ckpt)

We take the `genbio-ai/IsoformExpression-Enformer-AIDO.RNA-1.6B-CDS-ConcatFusion-rna-isoform-expression-ckpt` as an example to show how to download the model.

**Download model**
```
from huggingface_hub import snapshot_download
from pathlib import Path

model_name = "genbio-ai/IsoformExpression-Enformer-AIDO.RNA-1.6B-CDS-ConcatFusion-rna-isoform-expression-ckpt"
genbio_models_path = Path.home().joinpath('genbio_models', model_name)
genbio_models_path.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=model_name, local_dir=genbio_models_path)
```

Once you download the model and config file, you can refer to the above `evaluation script` for how to use the model for inference.


## Reference

[1] Multi-modal Transfer Learning between Biological Foundation Models. Garau-Luis et al., NeurIPS 2024.