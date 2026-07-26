# Copyright 2025 GenBio AI

set -xe

source activate msa

PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/..

export PATH=/workspace/env/mmseqs/bin/:$PATH

#input=${PROJECT_DIR}/data/example/  # input a folder, could include multiple .fasta files
input=${PROJECT_DIR}/data/example/T1104-D1.fasta  # input a fasta, could include multiple sequences
output_dir=local_msa_database/
mkdir -p ${output_dir}

#config_yaml_path=${PROJECT_DIR}/yamls/mmseqs.yaml
config_yaml_path=${PROJECT_DIR}/yamls/mmseqs_api.yaml

cpus_per_task=4
no_tasks=60  # tuning this number according to machine setting.

python ${PROJECT_DIR}/search_msa.py \
    --input=${input} \
    --output_dir=${output_dir} \
    --cpus_per_task=${cpus_per_task} \
    --no_tasks=${no_tasks} \
    --config_yaml_path=${config_yaml_path} \
    --shuffle_file_list \
