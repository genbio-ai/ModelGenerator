# Shuxian Zou
MODE=train
ORGANISM=hsapiens   #'athaliana' 'dmelanogaster' 'ecoli' 'hsapiens' 'scerevisiae'

PROJECT=rna_tasks
if [ $MODE == "train" ]; then
  for FOLD in {0..4}
  do
    RUN_NAME=pa_${ORGANISM}_aido_rna_1b600mـcds_fold${FOLD}
    CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
    mgen fit --config experiments/AIDO.RNA/configs/protein_abundance.yaml \
      --data.config_name protein_abundance_${ORGANISM} \
      --data.cv_test_fold_id $FOLD \
      --trainer.logger.name $RUN_NAME \
      --trainer.callbacks.dirpath $CKPT_SAVE_DIR
  done
else
  for FOLD in {0..4}
  do
    RUN_NAME=pa_${ORGANISM}_aido_rna_1b600mـcds_fold${FOLD}
    CKPT_PATH=logs/${PROJECT}/${RUN_NAME}/best_val*
    echo ">>> Fold ${FOLD}"
    mgen test --config experiments/AIDO.RNA/configs/protein_abundance.yaml \
      --data.config_name protein_abundance_${ORGANISM} \
      --data.cv_test_fold_id $FOLD \
      --model.strict_loading False \
      --model.reset_optimizer_states True \
      --trainer.logger null \
      --ckpt_path $CKPT_PATH
  done
fi
