# Shuxian Zou
MODE=train
CELL_LINE=Muscle  #'Muscle' 'HEK' 'pc3'

if [ $MODE == "train" ]; then
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
else
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
fi
