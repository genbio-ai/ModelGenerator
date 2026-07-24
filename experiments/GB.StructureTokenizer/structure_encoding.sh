# 1. download the sample dataset
huggingface-cli download genbio-ai/sample-structure-dataset --repo-type dataset --local-dir ./data/protstruct_sample_data/

set -ex

# 2. run encoding and then decoding
# check logs/protstruct_model/casp15_pdb_files
# *_input.pdb are the original pdb files
# *_output.pdb are the reconstructed pdb files
echo "run encoding and then decoding"
CUDA_VISIBLE_DEVICES=0 mgen predict --config=experiments/GB.StructureTokenizer/encode_decode.yaml


# 3. run encoding only
# check logs/protstruct_encode/casp15_struct_tokens.pt for the output tokens
# logs/protstruct_encode/codebook.pt for the codebook
echo "run encoding only"
CUDA_VISIBLE_DEVICES=0 mgen predict --config=experiments/GB.StructureTokenizer/encode.yaml

# 4. decode the tokens from step 3
# check logs/protstruct_decode/casp15_pdb_files for the output structures
echo "decode the tokens from the encoding step"
CUDA_VISIBLE_DEVICES=0 mgen predict --config=experiments/GB.StructureTokenizer/decode.yaml
