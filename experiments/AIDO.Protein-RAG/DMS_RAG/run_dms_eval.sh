#!/usr/bin/env bash
# set -x
set -e

SCRIPT_PATH=`dirname "$(realpath $0)"`

FOLDS=0,1,2,3,4
PROJECT=DMS_Benchmark
PRECISION=bf16-mixed
CONFIG=${SCRIPT_PATH}/configs/substitution_LoRA_DDP.yaml
STR_EMBEDDING_IN=384
RANDOM_ID=""
BACKBONE=null
NODE=localhost
POSITIONAL_ARGS=()

HELP="\x1b[0;33;49m$0 [--backbone backbone] [--folds 0,1,2,3,4] [--node a3mega-a3meganodeset-61] [--config config] [--mask-str] [--project DMS_Benchmark] [--precision 32] [--random-id 0nkspmxk] DMS_ID \x1b[0m"

while [[ $# -gt 0 ]]; do
  case $1 in
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
    --config)
      CONFIG="$2"
      shift
      shift
      ;;
    --node)
      NODE="$2"
      shift
      shift
      ;;
    --random-id)
      RANDOM_ID="$2"
      shift
      shift
      ;;
    -h|--help)
      echo -e $HELP
      exit 1
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo -e $HELP
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
    echo -e $HELP
    exit 1
fi

if [[ "$PRECISION" != "bf16-mixed" && "$PRECISION" != "bf16-true" && "$PRECISION" != "16-mixed" && "$PRECISION" != "16-true" && "$PRECISION" != "32" ]]; then
    echo "Invalid precision: $PRECISION. Should be one of bf16-mixed, bf16-true, 16-mixed, 16-true, 32"
    echo -e $HELP
    exit 1
fi

IFS=',' read -ra FOLDS <<< "$FOLDS"

echo -e "\x1b[0;33;49m============================================\x1b[0m"
echo -e "\x1b[0;33;49m FOLDS: ${FOLDS} \x1b[0m"
echo -e "\x1b[0;33;49m PROJECT: ${PROJECT} \x1b[0m"
echo -e "\x1b[0;33;49m PRECISION: ${PRECISION} \x1b[0m"
echo -e "\x1b[0;33;49m CONFIG: ${CONFIG} \x1b[0m"
echo -e "\x1b[0;33;49m RANDOM_ID: ${RANDOM_ID} \x1b[0m"
echo -e "\x1b[0;33;49m BACKBONE: ${BACKBONE} \x1b[0m"
echo -e "\x1b[0;33;49m TASK_NAME: ${TASK_NAME} \x1b[0m"
echo -e "\x1b[0;33;49m STR_EMBEDDING_IN: ${STR_EMBEDDING_IN} \x1b[0m"
echo -e "\x1b[0;33;49m NODE: ${NODE} \x1b[0m"
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

MUTATION_TYPE='singles_substitutions'

# echo ${FOLDS[@]}

mkdir -p ${SCRIPT_PATH}/DMS_output/${PROJECT}
mkdir -p ${SCRIPT_PATH}/output_logs/${PROJECT}

mgen_exec=`which mgen || echo ""`
torchrun_exec=`which torchrun || echo ""`
if [[ "$mgen_exec" == "" ]] || [[ "$torchrun_exec" == "" ]]; then
    echo -e "\x1b[0;31;49m Error: mgen or torchrun not in PATH \x1b[0m"
    exit 1;
fi

for FOLD in ${FOLDS[@]};
do
    # echo $FOLD
    RUN_NAME=${TASK_NAME}_fold${FOLD}

    if [[ "${RANDOM_ID}" == "" ]]; then
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
        --trainer.callbacks.output_dir ${SCRIPT_PATH}/DMS_output/${PROJECT}/${RUN_NAME} \
        --trainer.num_nodes 1 \
        --trainer.precision ${PRECISION} \
        --trainer.logger null \
        --model.strict_loading false \
        --data.batch_size 1 \
        --data.train_split_files [${MUTATION_TYPE}/${TASK_NAME}.tsv] \
        --data.cv_test_fold_id ${FOLD} \
        --ckpt_path ${CKPT_PATH}"
    
    if [[ "$BACKBONE" != "null" ]]; then
        cmd=$cmd" --model.init_args.backbone.class_path ${BACKBONE}"
    fi

    new_cmd=$(echo $cmd | tr -s ' ')
    echo -e "\x1b[0;94;49m ${new_cmd} \x1b[0m"

    ssh ${NODE} "source ${SCRIPT_PATH}/init_env.sh && ${new_cmd} &> ${SCRIPT_PATH}/output_logs/${PROJECT}/${RUN_NAME}_test.log";

done;

