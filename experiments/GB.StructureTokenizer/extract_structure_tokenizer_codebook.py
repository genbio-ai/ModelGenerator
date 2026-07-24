import torch
from modelgenerator.structure_tokenizer.models.structure_tokenizer import StructureTokenizerModel
import argparse


def main():
    """
    Main function to extract and save the codebook of the StructureTokenizerModel.

    The codebook is a matrix that represents the embeddings of structure tokens in the model.
    It has the shape (num_tokens, embedding_dim), where:
      - num_tokens: The number of unique structure tokens in the model's vocabulary.
      - embedding_dim: The dimensionality of the token embeddings.

    This script loads a pretrained StructureTokenizerModel, extracts its codebook, and saves
    it as a PyTorch tensor file (.pt).

    Usage:
      Run the script with the required arguments to specify the output file path and the
      pretrained model to use. The codebook will be saved in the specified path.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract the codebook of StructureTokenizerModel.\n"
            "The codebook is a matrix of shape (num_tokens, embedding_dim), where each row corresponds "
            "to the embedding of a structure token. The extracted codebook is saved as a PyTorch tensor (.pt) file."
        )
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the codebook in .pt format.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="genbio-ai/GB.StructureTokenizer",
        help=(
            "The pretrained model name or local path to load the StructureTokenizerModel.\n"
            "Default: 'genbio-ai/GB.StructureTokenizer'."
        ),
    )
    args = parser.parse_args()

    model = StructureTokenizerModel.from_pretrained(args.pretrained_model_name_or_path)
    codebook = model.encoder.codebook.data.cpu()
    # Save the extracted codebook as a PyTorch tensor (.pt) file
    torch.save(codebook, args.output_path)


if __name__ == "__main__":
    main()
