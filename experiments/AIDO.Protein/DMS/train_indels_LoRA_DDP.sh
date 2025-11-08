# Ning Sun
TASK_NAME='B1LPA6_ECOSM_Russ_2020_indels'
MUTATION_TYPE='indels'
for FOLD in {0..4}
do
RUN_NAME=${TASK_NAME}_fold${FOLD}
srun mgen fit --config experiments/AIDO.Protein/DMS/configs/indels_LoRA_DDP.yaml \
    --data.train_split_files "[\"${MUTATION_TYPE}/${TASK_NAME}.tsv\"]" \
    --trainer.logger.name ${RUN_NAME} \
    --trainer.logger.id ${RUN_NAME} \
    --data.cv_test_fold_id ${FOLD} \
    --trainer.num_nodes 2 \
    --data.batch_size 1 \
    --trainer.callbacks.patience 5 \
    &> output_logs/protein/${RUN_NAME}.log
done
