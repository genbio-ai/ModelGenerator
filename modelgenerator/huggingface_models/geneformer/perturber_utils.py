import logging
import pickle
import numpy as np
import pandas as pd
import torch

from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BitsAndBytesConfig,
)

from . import (
    TOKEN_DICTIONARY_FILE,
    ENSEMBL_DICTIONARY_FILE,
)

logger = logging.getLogger(__name__)


def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums)) + 1


def pad_list(input_ids, pad_token_id, max_len):
    input_ids = np.pad(
        input_ids,
        (0, max_len - len(input_ids)),
        mode="constant",
        constant_values=pad_token_id,
    )
    return input_ids


def pad_xd_tensor(tensor, pad_token_id, max_len, dim):
    padding_length = max_len - tensor.size()[dim]
    # Construct a padding configuration where all padding values are 0, except for the padding dimension
    # 2 * number of dimensions (padding before and after for every dimension)
    pad_config = [0] * 2 * tensor.dim()
    # Set the padding after the desired dimension to the calculated padding length
    pad_config[-2 * dim - 1] = padding_length
    return torch.nn.functional.pad(
        tensor, pad=pad_config, mode="constant", value=pad_token_id
    )


def pad_tensor(tensor, pad_token_id, max_len):
    tensor = torch.nn.functional.pad(
        tensor, pad=(0, max_len - tensor.numel()), mode="constant", value=pad_token_id
    )

    return tensor


def pad_2d_tensor(tensor, pad_token_id, max_len, dim):
    if dim == 0:
        pad = (0, 0, 0, max_len - tensor.size()[dim])
    elif dim == 1:
        pad = (0, max_len - tensor.size()[dim], 0, 0)
    tensor = torch.nn.functional.pad(
        tensor, pad=pad, mode="constant", value=pad_token_id
    )
    return tensor


def pad_3d_tensor(tensor, pad_token_id, max_len, dim):
    if dim == 0:
        raise Exception("dim 0 usually does not need to be padded.")
    if dim == 1:
        pad = (0, 0, 0, max_len - tensor.size()[dim])
    elif dim == 2:
        pad = (0, max_len - tensor.size()[dim], 0, 0)
    tensor = torch.nn.functional.pad(
        tensor, pad=pad, mode="constant", value=pad_token_id
    )
    return tensor


def pad_or_truncate_encoding(encoding, pad_token_id, max_len):
    if isinstance(encoding, torch.Tensor):
        encoding_len = encoding.size()[0]
    elif isinstance(encoding, list):
        encoding_len = len(encoding)
    if encoding_len > max_len:
        encoding = encoding[0:max_len]
    elif encoding_len < max_len:
        if isinstance(encoding, torch.Tensor):
            encoding = pad_tensor(encoding, pad_token_id, max_len)
        elif isinstance(encoding, list):
            encoding = pad_list(encoding, pad_token_id, max_len)
    return encoding


# pad list of tensors and convert to tensor
def pad_tensor_list(
    tensor_list,
    dynamic_or_constant,
    pad_token_id,
    model_input_size,
    dim=None,
    padding_func=None,
):
    # determine maximum tensor length
    if dynamic_or_constant == "dynamic":
        max_len = max([tensor.squeeze().numel() for tensor in tensor_list])
    elif isinstance(dynamic_or_constant, int):
        max_len = dynamic_or_constant
    else:
        max_len = model_input_size
        logger.warning(
            "If padding style is constant, must provide integer value. "
            f"Setting padding to max input size {model_input_size}."
        )

    # pad all tensors to maximum length
    if dim is None:
        tensor_list = [
            pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list
        ]
    else:
        tensor_list = [
            padding_func(tensor, pad_token_id, max_len, dim) for tensor in tensor_list
        ]
    # return stacked tensors
    if padding_func != pad_3d_tensor:
        return torch.stack(tensor_list)
    else:
        return torch.cat(tensor_list, 0)

