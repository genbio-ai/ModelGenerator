# Shuxian Zou
MODE=train
ORGANISM=ecoli   #'athaliana' 'dmelanogaster' 'ecoli' 'hsapiens' 'scerevisiae' 'ppastoris' 'hvolcanii'

PROJECT=rna_tasks
if [ $MODE == "train" ]; then
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
else
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
fi
