# Shuxian Zou
MODE=train

RUN_NAME=GB.Protein_16B_fsdp_bs4
PROJECT=xtrimo_benchmark
CONFIG_FILE=experiments/GB.Protein/xTrimo/configs/contact_prediction_binary.yaml
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}

if [ $MODE == "train" ]; then
  # using slurm script with 1 nodes (4 gpus in total) for training
  srun mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --trainer.num_nodes 1
else
  CKPT_PATH=${CKPT_SAVE_DIR}/best_val*.ckpt
  mgen test --config $CONFIG_FILE  \
    --data.batch_size 1 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
fi
