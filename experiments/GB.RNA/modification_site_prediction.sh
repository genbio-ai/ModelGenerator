# Shuxian Zou
MODE=train

RUN_NAME=msp_gb_rna_1b600m
PROJECT=rna_tasks
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CONFIG_FILE=experiments/GB.RNA/configs/modification_site_prediction.yaml

if [ $MODE == "train" ]; then
  # using slurm script with 4 nodes (16 gpus in total) for training
  srun mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --trainer.num_nodes 4
else
  CKPT_PATH=${CKPT_SAVE_DIR}/best_val*
  mgen test --config $CONFIG_FILE \
    --data.batch_size 16 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
fi
