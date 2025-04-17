#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

[[ "${DEBUG:-}" == "1" ]] && set -x

: "${LAYERNORM_TYPE:=fast_layernorm}"
: "${USE_DEEPSPEED_EVO_ATTTENTION:=true}"
: "${device_ids:=0,1,2,3}"
: "${master_port:=8803}"
: "${seed:=1234}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
CHECKPOINT_PATH="/nfs/model"
CCD_COMPONENTS="/nfs/ccd"
CCD_COMPONENTS_RDKIT="/nfs/ccd"

echo ${PROJECT_DIR}

: "${CHECKPOINT_PATH:?Environment variable CHECKPOINT_PATH is required}"
: "${CCD_COMPONENTS:?Environment variable CCD_COMPONENTS is required}"
: "${CCD_COMPONENTS_RDKIT:?Environment variable CCD_COMPONENTS_RDKIT is required}"

yaml_file_path="${PROJECT_DIR}/configs/inference_v0.1.yaml"
checkpoint_path="${CHECKPOINT_PATH}/fold49-v0.1.2.pt"
ccd_components_file="${CCD_COMPONENTS}/components.v20240608.cif"
ccd_components_rdkit_mol_file="${CCD_COMPONENTS_RDKIT}/components.v20240608.cif.rdkit_mol.pkl"
input_json_path="${PROJECT_DIR}/examples/example_built.json"
output_dir="./outputs/example-${seed}"

for f in "${yaml_file_path}" "${checkpoint_path}" "${ccd_components_file}" "${ccd_components_rdkit_mol_file}" "${input_json_path}"; do
    [[ -f "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done

mkdir -p "${output_dir}"

export LAYERNORM_TYPE
export USE_DEEPSPEED_EVO_ATTTENTION

CUDA_VISIBLE_DEVICES="${device_ids}" \
OMP_NUM_THREADS=1 \
torchrun --nnodes=1 --nproc_per_node=4 --master_port="${master_port}" \
    "${PROJECT_DIR}/runner/inference.py" \
    --yaml_file_path="${yaml_file_path}" \
    --checkpoint_path="${checkpoint_path}" \
    --ccd_components_file="${ccd_components_file}" \
    --ccd_components_rdkit_mol_file="${ccd_components_rdkit_mol_file}" \
    --seeds="${seed}" \
    --dump_dir="${output_dir}" \
    --input_json_path="${input_json_path}"
