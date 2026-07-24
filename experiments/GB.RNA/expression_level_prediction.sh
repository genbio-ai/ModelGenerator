# Shuxian Zou
MODE=train
CELL_LINE=pc3  #'Muscle' 'HEK' 'pc3'

if [ $MODE == "train" ]; then
  for FOLD in {0..9}
  do
    RUN_NAME=el_${CELL_LINE}_gb_rna_1b600m_fold${FOLD}
    CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}
    CUDA_VISIBLE_DEVICES=0 mgen fit --config experiments/GB.RNA/configs/expression_level.yaml \
      --data.config_name expression_${CELL_LINE} \
      --data.cv_test_fold_id $FOLD \
      --trainer.logger.name $RUN_NAME \
      --trainer.callbacks.dirpath $CKPT_SAVE_DIR
  done
else
  for FOLD in {0..9}
  do
    CKPT_PATH=logs/rna_tasks/el_${CELL_LINE}_gb_rna_1b600m_fold${FOLD}/best_val*
    echo ">>> Fold ${FOLD}"
    mgen test --config experiments/GB.RNA/configs/expression_level.yaml \
      --data.config_name expression_${CELL_LINE} \
      --data.cv_test_fold_id $FOLD \
      --model.strict_loading True \
      --model.reset_optimizer_states True \
      --trainer.logger null \
      --ckpt_path $CKPT_PATH
  done
fi
