# Ning Sun
TASK_NAME='A4GRB6_PSEAI_Chen_2020'
MUTATION_TYPE='singles_substitutions'
for FOLD in {0..4}
do
RUN_NAME=${TASK_NAME}_fold${FOLD}
srun mgen fit --config experiments/AIDO.Protein/DMS/configs/substitution_LoRA_DDP.yaml \
    --data.train_split_files "[\"${MUTATION_TYPE}/${TASK_NAME}.tsv\"]" \
    --trainer.logger.name ${RUN_NAME} \
    --trainer.logger.id ${RUN_NAME} \
    --data.cv_test_fold_id ${FOLD} \
    --trainer.num_nodes 4 \
    --data.batch_size 2 \
    --trainer.callbacks.patience 5 \
    &> output_logs/protein/${RUN_NAME}.log
done
