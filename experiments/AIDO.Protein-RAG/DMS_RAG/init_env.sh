SCRIPT_PATH=`dirname "$(realpath ${BASH_SOURCE[0]})"`

source ~/.bash_profile
# echo "source ~/.bash_profile"
eval init_conda_env
conda activate python3.11
lib_nvjitlink

# which torchrun
MG_PATH=$(realpath ${SCRIPT_PATH}/../../..)
export PYTHONPATH=${MG_PATH}:${PYTHONPATH}
export OMP_NUM_THREADS=1
# export HF_DATASETS_OFFLINE=1
# export PL_GLOBAL_SEED=0
# export TF_ENABLE_ONEDNN_OPTS=0