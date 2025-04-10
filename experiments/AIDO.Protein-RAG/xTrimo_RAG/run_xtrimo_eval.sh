#!/usr/bin/env bash
# set -x
set -e

SCRIPT_PATH=`dirname "$(realpath ${BASH_SOURCE[0]})"`

PROJECT=xTrimo_Benchmark
PRECISION=bf16-mixed
STR_EMBEDDING_IN=384
# CONFIG=${SCRIPT_PATH}/configs/DMS/substitution_LoRA_DDP.yaml
# MORE_ARGS=""
BACKBONE=null
NODE=localhost
RANDOM_ID=""
POSITIONAL_ARGS=()

HELP="\x1b[0;33;49m$0 [--backbone backbone] [--mask-str] [--project xTrimo_Benchmark] [--precision bf16-mixed] [--random-id 0nkspmxk] [--node localhost] config run_name \x1b[0m"

while [[ $# -gt 0 ]]; do
  case $1 in
    --backbone)
      BACKBONE="$2"
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
    --random-id)
      RANDOM_ID="$2"
      shift
      shift
      ;;
    --node)
      NODE="$2"
      shift
      shift
      ;;
    -h|--help)
      echo -e $HELP
      exit 1
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift
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

CONFIG=${POSITIONAL_ARGS[0]}
RUN_NAME=${POSITIONAL_ARGS[1]}
CONFIG=$(realpath $CONFIG)

echo -e "\x1b[0;33;49m============================================\x1b[0m"
echo -e "\x1b[0;33;49m PROJECT: ${PROJECT} \x1b[0m"
echo -e "\x1b[0;33;49m RUN_NAME: ${RUN_NAME} \x1b[0m"
echo -e "\x1b[0;33;49m PRECISION: ${PRECISION} \x1b[0m"
echo -e "\x1b[0;33;49m CONFIG: ${CONFIG} \x1b[0m"
echo -e "\x1b[0;33;49m NODE: ${NODE} \x1b[0m"
echo -e "\x1b[0;33;49m BACKBONE: ${BACKBONE} \x1b[0m"
echo -e "\x1b[0;33;49m RANDOM_ID: ${RANDOM_ID} \x1b[0m"
echo -e "\x1b[0;33;49m STR_EMBEDDING_IN: ${STR_EMBEDDING_IN} \x1b[0m"
echo -e "\x1b[0;33;49m============================================\x1b[0m"

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

mkdir -p ${SCRIPT_PATH}/xTrimo_output/${PROJECT}
mkdir -p ${SCRIPT_PATH}/output_logs/${PROJECT}

mgen_exec=`which mgen || echo ""`
torchrun_exec=`which torchrun || echo ""`
if [[ "$mgen_exec" == "" ]] || [[ "$torchrun_exec" == "" ]]; then
    echo -e "\x1b[0;31;49m Error: mgen or torchrun not in PATH \x1b[0m"
    exit 1;
fi

if [[ "${RANDOM_ID}" == "" ]]; then
    # logs/xTrimo_Benchmark/xTrimo_Benchmark/7wf52iaq/checkpoints/best_val:epoch=5-val_accuracy=0.678.ckpt
    CKPT_PATH_LIST=($(ls --color=never -d ${SCRIPT_PATH}/logs/${PROJECT}/${PROJECT}/${RUN_NAME}_*))
    if [[ ${#CKPT_PATH_LIST[@]} -ne 1 ]]; then
        echo -e "\x1b[0;31;49mError: Checkpoint path has no checkpoint or has multiple checkpoints\x1b[0m"
        echo ${CKPT_PATH_LIST[@]}
        exit 1
    fi
    CKPT_PATH=${CKPT_PATH_LIST[0]}
else
    CKPT_PATH=${SCRIPT_PATH}/logs/${PROJECT}/${PROJECT}/${RUN_NAME}_${RANDOM_ID}
fi

CKPT_LIST=($(ls --color=never "${CKPT_PATH}/checkpoints" | grep -v last.ckpt))
if [[ ${#CKPT_LIST[@]} -ne 1 ]]; then
    echo -e "\x1b[0;31;49mError: Checkpoint path has no checkpoint or has multiple checkpoints\x1b[0m"
    echo ${CKPT_LIST[@]}
    exit 1
fi
CKPT_PATH="${CKPT_PATH}/checkpoints/${CKPT_LIST[0]}"
echo -e "\x1b[0;32;49m CKPT_PATH: ${CKPT_PATH} \x1b[0m"
echo -e "\x1b[0;32;49m Check ${SCRIPT_PATH}/output_logs/${PROJECT}/${RUN_NAME}_test.log \x1b[0m"

cmd="${mgen_exec} test \
    --config ${CONFIG} \
    --config ${SCRIPT_PATH}/configs/prediction_writer.yaml \
    --trainer.callbacks.output_dir ${SCRIPT_PATH}/xTrimo_output/${PROJECT}/${RUN_NAME} \
    --trainer.num_nodes 1 \
    --trainer.precision ${PRECISION} \
    --trainer.logger null \
    --model.strict_loading false \
    --data.batch_size 1 \
    --model.init_args.backbone.init_args.config_overwrites.str_embedding_in ${STR_EMBEDDING_IN} \
    --ckpt_path ${CKPT_PATH}"

if [[ "$BACKBONE" != "null" ]]; then
    cmd=$cmd" --model.init_args.backbone.class_path ${BACKBONE}"
fi

new_cmd=$(echo $cmd | tr -s ' ')
echo -e "\x1b[0;94;49m ${new_cmd} \x1b[0m"

ssh ${NODE} "source ${SCRIPT_PATH}/init_env.sh && ${new_cmd} &> ${SCRIPT_PATH}/output_logs/${PROJECT}/${RUN_NAME}_test.log"

# --model.reset_optimizer_states True

