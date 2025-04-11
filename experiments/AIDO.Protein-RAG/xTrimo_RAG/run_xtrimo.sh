#!/usr/bin/env bash
# set -x
set -e

SCRIPT_PATH=`dirname "$(realpath ${BASH_SOURCE[0]})"`

# export OMP_NUM_THREADS=1
# export HF_DATASETS_OFFLINE=1
# export PL_GLOBAL_SEED=0

PROJECT=xTrimo_Benchmark
# RUN_NAME=fold_AIDO.Protein.RAGPLM.w.structure
PRECISION=bf16-mixed
STR_EMBEDDING_IN=384
ACCUMULATE_GRAD_BATCHES=1
NODES=localhost
LR=null
GRAD_CLIP_VAL=null
RANDOM_ID=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
BACKBONE=null
DRY_RUN=false
MAX_LENGTH=12800
# CONFIG=${SCRIPT_PATH}/configs/fold_prediction.yaml
POSITIONAL_ARGS=()

HELP="\x1b[0;33;49m$0 [--backbone backbone] [--accumulate-grad-batches 1] [--mask-str] [--project xTrimo_Benchmark] [--precision bf16-mixed] [--lr 0.0001] [--grad-clip-val 0.1] [--random-id 0nkspmxk] [--max-length 12800] [--nodes a3mega-a3meganodeset-19,a3mega-a3meganodeset-20] [--dry-run] config run_name \x1b[0m"

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
    --mask-str)
      STR_EMBEDDING_IN=0
      shift
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift
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
    --accumulate-grad-batches)
      ACCUMULATE_GRAD_BATCHES="$2"
      shift
      shift
      ;;
    --config)
      CONFIG=$(realpath $2)
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
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift
      # echo -e ${HELP}
      # exit 1
      ;;
  esac
done

if [[ "$PRECISION" != "bf16-mixed" && "$PRECISION" != "bf16-true" && "$PRECISION" != "16-mixed" && "$PRECISION" != "16-true" && "$PRECISION" != "32" ]]; then
    echo "Invalid precision: $PRECISION. Should be one of bf16-mixed, bf16-true, 16-mixed, 16-true, 32"
    echo -e ${HELP}
    exit 1
fi

if [[ ${#POSITIONAL_ARGS[@]} -ne 2 ]]; then
    echo -e $HELP
    exit 1;
fi

if [[ ! -f ${SCRIPT_PATH}/init_env.sh ]]; then
    echo -e "\x1b[0;31;49m Error: ${SCRIPT_PATH}/init_env.sh not exists. \x1b[0m"
    exit 1;
fi

CONFIG=${POSITIONAL_ARGS[0]}
RUN_NAME=${POSITIONAL_ARGS[1]}
CONFIG=$(realpath $CONFIG)

echo -e "\x1b[0;33;49m============================================\x1b[0m"
echo -e "\x1b[0;33;49m PROJECT: ${PROJECT} \x1b[0m"
echo -e "\x1b[0;33;49m RUN_NAME: ${RUN_NAME} \x1b[0m"
echo -e "\x1b[0;33;49m PRECISION: ${PRECISION} \x1b[0m"
echo -e "\x1b[0;33;49m LR: ${LR} \x1b[0m"
echo -e "\x1b[0;33;49m GRAD_CLIP_VAL: ${GRAD_CLIP_VAL} \x1b[0m"
echo -e "\x1b[0;33;49m ACCUMULATE_GRAD_BATCHES: ${ACCUMULATE_GRAD_BATCHES} \x1b[0m"
echo -e "\x1b[0;33;49m CONFIG: ${CONFIG} \x1b[0m"
echo -e "\x1b[0;33;49m RANDOM_ID: ${RANDOM_ID} \x1b[0m"
echo -e "\x1b[0;33;49m NODES: ${NODES[@]} \x1b[0m"
echo -e "\x1b[0;33;49m BACKBONE: ${BACKBONE} \x1b[0m"
echo -e "\x1b[0;33;49m MAX_LENGTH: ${MAX_LENGTH} \x1b[0m"
echo -e "\x1b[0;33;49m DRY_RUN: ${DRY_RUN} \x1b[0m"
echo -e "\x1b[0;33;49m STR_EMBEDDING_IN: ${STR_EMBEDDING_IN} \x1b[0m"
echo -e "\x1b[0;33;49m============================================\x1b[0m"

IFS=',' read -ra NODES <<< "$NODES"
NNODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
MASTER_PORT=$(( RANDOM % (50000 - 10000 + 1) + 10000 ))

if [[ "$BACKBONE" != "null" ]]; then
    if [[ "$BACKBONE" != "modelgenerator.backbones.aido_protein_rag_3b" ]] && [[ "$BACKBONE" != "modelgenerator.backbones.aido_protein_rag_16b" ]] && [[ "$BACKBONE" != "modelgenerator.backbones.aido_protein_3b" ]];
    then
        echo -e "\x1b[0;31;49m Error: --backbone only support modelgenerator.backbones.aido_protein_rag_3b, modelgenerator.backbones.aido_protein_rag_16b and modelgenerator.backbones.aido_protein_3b \x1b[0m"
        exit 1;
    fi

    if [[ "$BACKBONE" == "modelgenerator.backbones.aido_protein_rag_3b" ]] && [[ "$STR_EMBEDDING_IN" != "0" ]]; then
        echo -e "--mask-str must be specified when use modelgenerator.backbones.aido_protein_rag_3b as backbone"
        exit 1;
    fi

    if [[ "$BACKBONE" == "modelgenerator.backbones.aido_protein_3b" ]] && [[ "$STR_EMBEDDING_IN" != "0" ]]; then
        echo -e "--mask-str must be specified when use modelgenerator.backbones.aido_protein_3b as backbone"
        exit 1;
    fi
fi

mkdir -p ${SCRIPT_PATH}/output_logs/${PROJECT}
mkdir -p ${SCRIPT_PATH}/logs/${PROJECT}/wandb
mkdir -p ${SCRIPT_PATH}/logs/${PROJECT}/${PROJECT}

mgen_exec=`which mgen || echo ""`
torchrun_exec=`which torchrun || echo ""`
if [[ "$mgen_exec" == "" ]] || [[ "$torchrun_exec" == "" ]]; then
    echo -e "\x1b[0;31;49m Error: mgen or torchrun not in PATH \x1b[0m"
    exit 1;
fi

echo -e "\x1b[0;32;49m ${RUN_NAME} \x1b[0m"
echo -e "\x1b[0;32;49m Check ${SCRIPT_PATH}/output_logs/${PROJECT}/${RUN_NAME}_Rank0.log \x1b[0m"

pids=()
NODE_RANK=0
for NODE in ${NODES[@]};
do
    # --trainer.callbacks.dirpath ${SCRIPT_PATH}/logs/${PROJECT}/${RUN_NAME} \
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
            --trainer.logger.save_dir ${SCRIPT_PATH}/logs/${PROJECT} \
            --trainer.default_root_dir ${SCRIPT_PATH}/logs/${PROJECT} \
            --trainer.precision ${PRECISION} \
            --trainer.accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES} \
            --trainer.num_nodes ${NNODES} \
            --model.init_args.backbone.init_args.max_length ${MAX_LENGTH} \
            --data.init_args.max_context_length ${MAX_LENGTH} \
            --model.init_args.backbone.init_args.config_overwrites.str_embedding_in ${STR_EMBEDDING_IN} \
            --data.batch_size 1"

    if [[ "$BACKBONE" != "null" ]]; then
        cmd=$cmd" --model.init_args.backbone.class_path ${BACKBONE}"
    fi

    if [[ "$LR" != "null" ]]; then
        cmd=$cmd" --model.init_args.optimizer.init_args.lr ${LR}"
    fi

    if [[ "${GRAD_CLIP_VAL}" != "null" ]]; then
        cmd=$cmd" --trainer.gradient_clip_val ${GRAD_CLIP_VAL}"
    fi

    echo -e "====== NODE: ${NODE}; NODE_RANK: ${NODE_RANK} ======"
    new_cmd=$(echo $cmd | tr -s ' ')
    echo -e "\x1b[0;94;49m ${new_cmd} \x1b[0m"

    if [[ "$DRY_RUN" == "false" ]]; then
        ssh ${NODE} "source ${SCRIPT_PATH}/init_env.sh && ${new_cmd} &> ${SCRIPT_PATH}/output_logs/${PROJECT}/${RUN_NAME}_Rank${NODE_RANK}.log" &
        pids+=($!);
    fi
    NODE_RANK=$(( NODE_RANK + 1 ));
done

wait "${pids[@]}"
