# Protein Structure Tokenizer
This is the implementation for [genbio-ai/GB.StructureTokenizer](https://huggingface.co/genbio-ai/GB.StructureTokenizer). Due to the properties of protein data, it has a standalone data pipeline. The overall structure of this folder is as follows:

- `callbacks`: Contains the callbacks used in saving structure tokens and PDB files
- `configs`: Contains the configuration files for the model and data
- `datasets`: Contains the dataset classes for handling PDB data and the data module
- `layers`: Contains the custom layers used in the model
- `models`: Contains the encoder (`equiformer_encoder.py`), decoder (`esmfold_decoder.py`), the full model (`structure_tokenizer.py`), and its lightning module (`structure_tokenizer_lightning.py`)
- `utils`: Miscellaneous utility functions

For the usage of this model, please refer to `experiments/GB.StructureTokenizer/README.md`.