# Shuxian Zou
MODE=train

RUN_NAME=tcr_pmhc_affinity_AIDO.Protein_16B
PROJECT=xtrimo_benchmark
CKPT_SAVE_DIR=logs/${PROJECT}/${RUN_NAME}
CONFIG_FILE=experiments/AIDO.Protein/xTrimo/configs/tcr_pmhc_affinity.yaml

if [ $MODE == "train" ]; then
  # using slurm script with 2 nodes (8 gpus in total) for training
  srun mgen fit --config $CONFIG_FILE \
    --trainer.logger.name $RUN_NAME \
    --trainer.logger.project $PROJECT \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR \
    --trainer.num_nodes 2
else
  CKPT_PATH=${CKPT_SAVE_DIR}/best_val*
  mgen test --config $CONFIG_FILE \
    --data.batch_size 16 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
fi
