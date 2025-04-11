#!/usr/bin/env bash
# set -x
set -e

SCRIPT_PATH=`dirname "$(realpath ${BASH_SOURCE[0]})"`

# export OMP_NUM_THREADS=1
# export HF_DATASETS_OFFLINE=1
# export PL_GLOBAL_SEED=0

FOLDS=0,1,2,3,4
PROJECT=DMS_Benchmark
PRECISION=bf16-mixed
STR_EMBEDDING_IN=384
PATIENCE=10
ACCUMULATE_GRAD_BATCHES=1
NODES=localhost
LR=null
GRAD_CLIP_VAL=null
DROPOUT=null
BACKBONE=null
DRY_RUN=false
CONFIG=${SCRIPT_PATH}/configs/substitution_LoRA_DDP.yaml
RANDOM_ID=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
POSITIONAL_ARGS=()

HELP="\x1b[0;33;49m$0 [--backbone backbone] [--folds 0,1,2,3,4] [--config config] [--accumulate-grad-batches 1] [--mask-str] [--project DMS_Benchmark] [--precision bf16-mixed] [--patience 10] [--random-id 0nkspmxk] [--lr 0.0001] [--grad-clip-val 0.1] [--dropout 0.1] [--nodes a3mega-a3meganodeset-19,a3mega-a3meganodeset-20] [--dry-run] DMS_ID \x1b[0m"

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --backbone)
      BACKBONE="$2"
      shift
      shift
      ;;
    --folds)
      FOLDS="$2"
      shift
      shift
      ;;
    --mask-str)
      STR_EMBEDDING_IN=0
      shift
      ;;
    --project)
      PROJECT="$2"
      shift
      shift
      ;;
    --precision)
      PRECISION="$2"
      shift
      shift
      ;;
    --lr)
      LR="$2"
      shift
      shift
      ;;
    --grad-clip-val)
      GRAD_CLIP_VAL="$2"
      shift
      shift
      ;;
    --dropout)
      DROPOUT="$2"
      shift
      shift
      ;;
    --accumulate-grad-batches)
      ACCUMULATE_GRAD_BATCHES="$2"
      shift
      shift
      ;;
    --patience)
      PATIENCE="$2"
      shift
      shift
      ;;
    --config)
      CONFIG="$2"
      shift
      shift
      ;;
    --random-id)
      RANDOM_ID="$2"
      shift
      shift
      ;;
    --nodes)
      NODES="$2"
      shift
      shift
      ;;
    -h|--help)
      echo -e ${HELP}
      exit 1
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo -e ${HELP}
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift
      ;;
  esac
done

TASK_NAME=${POSITIONAL_ARGS[0]}

if [[ -z "$TASK_NAME" ]]; then
    echo "Task name is required"
    echo -e ${HELP}
    exit 1
fi

if [[ "$PRECISION" != "bf16-mixed" && "$PRECISION" != "bf16-true" && "$PRECISION" != "16-mixed" && "$PRECISION" != "16-true" && "$PRECISION" != "32" ]]; then
    echo "Invalid precision: $PRECISION. Should be one of bf16-mixed, bf16-true, 16-mixed, 16-true, 32"
    echo -e ${HELP}
    exit 1
fi

echo -e "\x1b[0;33;49m============================================\x1b[0m"
echo -e "\x1b[0;33;49m FOLDS: ${FOLDS} \x1b[0m"
echo -e "\x1b[0;33;49m PROJECT: ${PROJECT} \x1b[0m"
echo -e "\x1b[0;33;49m PRECISION: ${PRECISION} \x1b[0m"
echo -e "\x1b[0;33;49m PATIENCE: ${PATIENCE} \x1b[0m"
echo -e "\x1b[0;33;49m ACCUMULATE_GRAD_BATCHES: ${ACCUMULATE_GRAD_BATCHES} \x1b[0m"
echo -e "\x1b[0;33;49m CONFIG: ${CONFIG} \x1b[0m"
echo -e "\x1b[0;33;49m RANDOM_ID: ${RANDOM_ID} \x1b[0m"
echo -e "\x1b[0;33;49m TASK_NAME: ${TASK_NAME} \x1b[0m"
echo -e "\x1b[0;33;49m BACKBONE: ${BACKBONE} \x1b[0m"
echo -e "\x1b[0;33;49m STR_EMBEDDING_IN: ${STR_EMBEDDING_IN} \x1b[0m"
echo -e "\x1b[0;33;49m NODES: ${NODES} \x1b[0m"
echo -e "\x1b[0;33;49m LR: ${LR} \x1b[0m"
echo -e "\x1b[0;33;49m DRY_RUN: ${DRY_RUN} \x1b[0m"
echo -e "\x1b[0;33;49m GRAD_CLIP_VAL: ${GRAD_CLIP_VAL} \x1b[0m"
echo -e "\x1b[0;33;49m DROPOUT: ${DROPOUT} \x1b[0m"
echo -e "\x1b[0;33;49m============================================\x1b[0m"

IFS=',' read -ra FOLDS <<< "$FOLDS"
IFS=',' read -ra NODES <<< "$NODES"
NNODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
MASTER_PORT=$(( RANDOM % (50000 - 10000 + 1) + 10000 ))

if [[ "$BACKBONE" != "null" ]]; then
    if [[ "$BACKBONE" != "modelgenerator.backbones.aido_protein_rag_3b" ]] && [[ "$BACKBONE" != "modelgenerator.backbones.aido_protein_rag_16b" ]];
    then
        echo -e "\x1b[0;31;49m Error: --backbone only support modelgenerator.backbones.aido_protein_rag_3b or modelgenerator.backbones.aido_protein_rag_16b \x1b[0m"
        exit 1;
    fi

    if [[ "$BACKBONE" == "modelgenerator.backbones.aido_protein_rag_3b" ]] && [[ "$STR_EMBEDDING_IN" != "0" ]]; then
        echo -e "--mask-str must be specified when use modelgenerator.backbones.aido_protein_rag_3b as backbone"
        exit 1;
    fi
fi

MUTATION_TYPE='singles_substitutions'

mkdir -p ${SCRIPT_PATH}/output_logs/${PROJECT}
mkdir -p ${SCRIPT_PATH}/logs/${PROJECT}/wandb
mkdir -p ${SCRIPT_PATH}/logs/${PROJECT}/${PROJECT}

mgen_exec=`which mgen || echo ""`
torchrun_exec=`which torchrun || echo ""`
if [[ "$mgen_exec" == "" ]] || [[ "$torchrun_exec" == "" ]]; then
    echo -e "\x1b[0;31;49m Error: mgen or torchrun not in PATH \x1b[0m"
    exit 1;
fi

for FOLD in ${FOLDS[@]};
do
    RUN_NAME=${TASK_NAME}_fold${FOLD}

    echo -e "\x1b[0;32;49m ${RUN_NAME}_${RANDOM_ID} \x1b[0m"
    echo -e "\x1b[0;32;49m Check ${SCRIPT_PATH}/output_logs/${PROJECT}/${RUN_NAME}_Rank0.log \x1b[0m"

    pids=()
    NODE_RANK=0
    for NODE in ${NODES[@]};
    do
        # bash ${SCRIPT_PATH}/init_env.sh;

        cmd="torchrun \
            --nproc_per_node=8 \
            --nnodes      ${NNODES} \
            --node_rank   ${NODE_RANK} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT} \
            ${mgen_exec} fit \
                --config ${CONFIG} \
                --config ${SCRIPT_PATH}/configs/wandb.yaml \
                --trainer.logger.project ${PROJECT} \
                --trainer.logger.name ${RUN_NAME} \
                --trainer.logger.id ${RUN_NAME}_${RANDOM_ID} \
                --trainer.default_root_dir ${SCRIPT_PATH}/logs/${PROJECT} \
                --trainer.logger.save_dir ${SCRIPT_PATH}/logs/${PROJECT} \
                --data.train_split_files [${MUTATION_TYPE}/${TASK_NAME}.tsv] \
                --data.cv_test_fold_id ${FOLD} \
                --trainer.precision ${PRECISION} \
                --trainer.accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES} \
                --trainer.num_nodes ${NNODES} \
                --data.batch_size 1 \
                --model.init_args.backbone.init_args.config_overwrites.str_embedding_in ${STR_EMBEDDING_IN} \
                --trainer.callbacks.patience ${PATIENCE}"
        
        if [[ "$BACKBONE" != "null" ]]; then
            cmd=$cmd" --model.init_args.backbone.class_path ${BACKBONE}"
        fi

        if [[ "$LR" != "null" ]]; then
            cmd=$cmd" --model.init_args.optimizer.init_args.lr ${LR}"
        fi

        if [[ "${GRAD_CLIP_VAL}" != "null" ]]; then
            cmd=$cmd" --trainer.gradient_clip_val ${GRAD_CLIP_VAL}"
        fi

        if [[ "${DROPOUT}" != "null" ]]; then
            cmd=$cmd" --model.init_args.adapter.init_args.dropout ${DROPOUT}"
            cmd=$cmd" --model.init_args.backbone.init_args.lora_dropout ${DROPOUT}"
        fi

        echo -e "====== NODE: ${NODE}; NODE_RANK: ${NODE_RANK} ======"
        new_cmd=$(echo $cmd | tr -s ' ')
        echo -e "\x1b[0;94;49m ${new_cmd} \x1b[0m"

        if [[ "$DRY_RUN" == "false" ]]; then
            ssh ${NODE} "source ${SCRIPT_PATH}/init_env.sh && ${new_cmd} &> ${SCRIPT_PATH}/output_logs/${PROJECT}/${RUN_NAME}_Rank${NODE_RANK}.log" &
            pids+=($!)
        fi
        NODE_RANK=$(( NODE_RANK + 1 ))
    done
    wait "${pids[@]}"
    MASTER_PORT=$(( MASTER_PORT + 1 ))

done
