set -e

DATA_DIR=data ## Set the path of the directory where you want to keep/download the PDB/CIF file.

PDB_ID=5YH2
CHAIN_ID=A

# ### Download and merge the Protein-IF checkpoint
# mkdir -p ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/
# ## Download chunks
# huggingface-cli download genbio-ai/AIDO.ProteinIF-16B \
# --repo-type model \
# --local-dir ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/
# ## Merge chunks
# python merge_ckpt.py ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model_chunks ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model.ckpt

### Download a single structure from somewhere like PDB
mkdir -p ${DATA_DIR}/
wget https://files.rcsb.org/download/${PDB_ID}.cif -P ${DATA_DIR}/

### Put it into our format
python preprocess_PDB.py ${DATA_DIR}/${PDB_ID}.cif ${CHAIN_ID} ${DATA_DIR}/

### Run inference to generate sequence
# export CUDA_VISIBLE_DEVICES=6,
mgen test \
    --config protein_inv_fold_test.yaml \
    --trainer.default_root_dir ${MGEN_DATA_DIR}/modelgenerator/logs/protein_inv_fold/ \
    --ckpt_path ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model.ckpt \
    --trainer.devices 0, \
    --data.path ${DATA_DIR}/

### The results will be saved under the folder "/experiments/AIDO.Protein/protein_inverse_folding/proteinIF_outputs" in a file named "results_acc_{recovery_accuracy}.txt". 
