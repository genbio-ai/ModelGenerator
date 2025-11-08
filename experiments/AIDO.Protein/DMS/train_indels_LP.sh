# Ning Sun
TASK_NAME='FECA_ECOLI_Tsuboyama_2023_2D1U_indels'
MUTATION_TYPE='indels'
for FOLD in {0..4}
do
RUN_NAME=${TASK_NAME}_fold${FOLD}
srun mgen fit --config experiments/AIDO.Protein/DMS/configs/indels_LP_DDP.yaml \
    --data.train_split_files "[\"${MUTATION_TYPE}/${TASK_NAME}.tsv\"]" \
    --trainer.logger.name ${RUN_NAME} \
    --trainer.logger.id ${RUN_NAME} \
    --data.cv_test_fold_id ${FOLD} \
    --trainer.num_nodes 1 \
    --trainer.devices 1 \
    --data.batch_size 8 \
    &> output_logs/protein/${RUN_NAME}.log
done
