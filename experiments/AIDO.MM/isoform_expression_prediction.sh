MODE=train

RUN_NAME=enformer_rnafm1.6b-cds_esm2_concat_fusion
CONFIG_FILE=experiments/AIDO.MM/configs/isoform_expression_dna_rna_prot_concat.yaml

# RUN_NAME=enformer_rnafm1.6b-cds_concat_fusion
# CONFIG_FILE=experiments/AIDO.MM/configs/isoform_expression_dna_rna_concat.yaml

# RUN_NAME=enformer_aidorna650m_esm2_attention_fusion
# CONFIG_FILE=experiments/AIDO.MM/configs/isoform_expression_dna_rna_prot_attention.yaml

# RUN_NAME=enformer_aidorna650m_attention_fusion
# CONFIG_FILE=experiments/AIDO.MM/configs/isoform_expression_dna_rna_attention.yaml

PROJECT=isoform_tasks
CKPT_SAVE_DIR=${GENBIO_DATA_DIR}/genbio_finetune/logs/${PROJECT}/${RUN_NAME}

if [ $MODE == "train" ]; then
  mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --model.optimizer.lr 1e-4 \
    --data.batch_size 1
else
  CKPT_PATH=${CKPT_SAVE_DIR}/best_val*
  mgen test --config $CONFIG_FILE \
      --data.batch_size 16 \
      --trainer.logger null \
      --model.strict_loading False \
      --model.reset_optimizer_states True \
      --ckpt_path $CKPT_PATH
fi
