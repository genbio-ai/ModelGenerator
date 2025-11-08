# usage: bash experiments/AIDO.StructureTokenizer/structure_prediction.sh
# need to run in the root directory of the project

set -ex

# 0. clean the tsv files in the output directory if they exist
# since we gather the partial tsv results in the output directory,
# we need to clean the output directory before running the script

# clean the tsv files in the output directory
rm -f logs/protein2structoken_16b/*.tsv

# 1. run the 16b language model to predict the structure tokens from protein sequences (amino acids)
# the input is the casp14, casp15, and cameo amino acid sequences (specified in the config file)
# the output is logs/protein2structoken_16b/predict_predictions.tsv
mgen predict --config experiments/AIDO.StructureTokenizer/protein2structoken_16b.yaml

# Alternatively, you can run the following command to specify your own input sequences
# mgen predict --config experiments/AIDO.StructureTokenizer/protein2structoken_16b.yaml \
#  --data.init_args.path=experiments/AIDO.StructureTokenizer/ \
#  --data.init_args.test_split_files=[protein2structoken_example_input.csv]

# 2. convert the predicted structures in tsv into one pt file
python experiments/AIDO.StructureTokenizer/struct_token_format_conversion.py logs/protein2structoken_16b/predict_predictions.tsv logs/protein2structoken_16b/predict_predictions.pt

# 3. extract the codebook of the structure tokenizer
# the output is logs/protein2structoken_16b/codebook.pt
python experiments/AIDO.StructureTokenizer/extract_structure_tokenizer_codebook.py --output_path logs/protein2structoken_16b/codebook.pt

# 4. run the decode model to convert the structure tokens into pdb files
# currently this script doesn't support multi-gpu, so only use one gpu
# the command line overrides the name, input structure tokens, and codebook path
# the output is logs/protstruct_decode/protein2structoken_16b_pdb_files/*output.pdb
CUDA_VISIBLE_DEVICES=0 mgen predict --config experiments/AIDO.StructureTokenizer/decode.yaml \
 --data.init_args.config.struct_tokens_datasets_configs.name=protein2structoken_16b \
 --data.init_args.config.struct_tokens_datasets_configs.struct_tokens_path=logs/protein2structoken_16b/predict_predictions.pt \
 --data.init_args.config.struct_tokens_datasets_configs.codebook_path=logs/protein2structoken_16b/codebook.pt
