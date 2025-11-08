# Shuxian Zou
MODE=train
SPLICE_SITE=acceptor  #'acceptor' 'donor'

RUN_NAME=csp_${SPLICE_SITE}_aido_rna_1b600m
if [ $MODE == "train" ]; then
  CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}
  CUDA_VISIBLE_DEVICES=0,1 mgen fit --config experiments/AIDO.RNA/configs/splice_site_prediction.yaml \
    --data.config_name splice_site_${SPLICE_SITE} \
    --trainer.logger.name $RUN_NAME \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR
else
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
      --ckpt_path $CKPT_PATH
  done
fi
