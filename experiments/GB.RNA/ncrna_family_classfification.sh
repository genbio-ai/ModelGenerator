# Shuxian Zou
MODE=train
BOUNDARY_NOISE=bnoise0   #'bnoise0' 'bnoise200'

RUN_NAME=nfc_${BOUNDARY_NOISE}_gb_rna_1b600m
if [ $MODE == "train" ]; then
  CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}
  CUDA_VISIBLE_DEVICES=0,1 mgen fit --config experiments/GB.RNA/configs/ncrna_family_classification.yaml \
    --data.config_name ncrna_family_${BOUNDARY_NOISE} \
    --trainer.callbacks.dirpath $CKPT_SAVE_DIR
else
  CKPT_PATH=logs/rna_tasks/${RUN_NAME}/best_val*
  mgen test --config experiments/GB.RNA/configs/ncrna_family_classification.yaml \
    --data.config_name ncrna_family_${BOUNDARY_NOISE} \
    --data.batch_size 256 \
    --model.strict_loading False \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
fi
