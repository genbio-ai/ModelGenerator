import os
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from lightning import LightningModule
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from lightning.pytorch.callbacks import Callback, BaseFinetuning
import glob
import pandas as pd
from typing import Optional, List, Any, Literal
from pathlib import Path
import re
import yaml


def replace_key_value(d: dict, key, value):
    # non-inplace replacement
    d = d.copy()
    d[key] = value
    return d


class PredictionWriter(Callback):
    """Write batch predictions to files, and merge batch files into a single file at the end of the epoch.
    Note:
        When saving the given data to a TSV file, any tensors in the data will have their last dimension
        squeezed and converted into lists to ensure proper formatting for TSV output.

    Args:
        output_dir (str): Directory to save predictions.
        filetype (str): Type of outputfile. Options are 'tsv' and 'pt'.
        write_cols (list): The head columns of tsv file if filetype is set to 'tsv'. Defaults to None
    """

    def __init__(
        self,
        output_dir: str,
        filetype: str,
        write_cols: Optional[List[str]] = None,
        drop_special_tokens: bool = False,  # doesn't work for pt filetype
        argmax_predictions: bool = False,  # run argmax on predictions before saving
        remove_duplicates: bool = False,  # remove duplicates in the final file, needs uid in the write_cols
        delete_intermediate_files: bool = False 
        # TODO: we need something that cleans up the intermediate files
    ):
        super().__init__()
        assert filetype in ["tsv", "pt"], "Only support tsv and pt filetype for now."
        assert (
            drop_special_tokens == False or filetype == "tsv"
        ), "drop_special_tokens only works for tsv filetype."
        assert (
            remove_duplicates == False or "uid" in write_cols
        ), "remove_duplicates requires 'uid' in write_cols."
        assert (
            remove_duplicates == False or filetype == "tsv"
        ), "remove_duplicates only works for tsv filetype."
        assert (
            delete_intermediate_files == False or remove_duplicates == True
        ), "delete_intermediate_files only works for remove_duplicates == True."

        self.output_dir = output_dir
        self.filetype = filetype
        self.write_cols = write_cols
        self.drop_special_tokens = drop_special_tokens
        self.argmax_predictions = argmax_predictions
        self.remove_duplicates = remove_duplicates
        self.delete_intermediate_files = delete_intermediate_files
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_batch(self, trainer, predictions, batch_idx, stage):
        if self.argmax_predictions:
            argmaxed = torch.argmax(predictions["predictions"], dim=-1)
            predictions = replace_key_value(predictions, "predictions", argmaxed)

        if self.drop_special_tokens:
            # TODO: this may output a lot of warnings, need to handle it
            if "attention_mask" not in predictions:
                rank_zero_warn(
                    "drop_special_tokens is set to True, but the model outputs do not contain 'attention_mask'."
                )
            elif "special_tokens_mask" not in predictions:
                rank_zero_warn(
                    "drop_special_tokens is set to True, but the model outputs do not contain 'special_tokens_mask'."
                )
            elif "predictions" not in predictions:
                rank_zero_warn(
                    "drop_special_tokens is set to True, but the model outputs do not contain 'predictions'."
                )
            elif (
                predictions["attention_mask"].shape
                != predictions["predictions"].shape[
                    : len(predictions["attention_mask"].shape)
                ]
            ):
                rank_zero_warn(
                    "drop_special_tokens is set to True, but the shape of 'attention_mask' does not match the shape of 'predictions'."
                    f"attention_mask shape: {predictions['attention_mask'].shape}, predictions shape: {predictions['predictions'].shape}"
                )
            else:
                mask = (
                    torch.tensor(predictions["attention_mask"]).bool().cpu()
                    & ~(torch.tensor(predictions["special_tokens_mask"]).bool().cpu())
                )
                new_predictions = {}
                predictions_no_special = [
                    pred[mask[i]].tolist()
                    for i, pred in enumerate(predictions["predictions"])
                ]
                # this is a non-inplace replacement
                # NOTE: the predictions now become a list of lists
                predictions = replace_key_value(
                    predictions, "predictions", predictions_no_special
                )

        if self.write_cols is None:
            save_predictions = predictions
        else:
            save_predictions = {key: predictions[key] for key in self.write_cols}

        if self.filetype == "tsv":
            # convert tensor to list to support tsv format saving
            for k in save_predictions.keys():
                if isinstance(save_predictions[k], torch.Tensor):
                    # << Keep same length for each item
                    v = save_predictions[k].squeeze(-1).tolist()
                    if isinstance(v, (tuple, list)) and len(v) != 1:
                        v = [v]
                    save_predictions[k] = v
            
            df = pd.DataFrame.from_dict(save_predictions)
            df.to_csv(
                os.path.join(
                    self.output_dir,
                    f"{stage}_predictions_{trainer.global_rank}_{batch_idx}.tsv",
                ),
                index=False,
                sep="\t",
            )

        elif self.filetype == "pt":
            torch.save(
                save_predictions,
                os.path.join(
                    self.output_dir,
                    f"{stage}_predictions_{trainer.global_rank}_{batch_idx}.pt",
                ),
            )
        else:
            ValueError(f"filetype {self.filetype} does not support")

    def _deduplicate_uid(self, df: pd.DataFrame):
        df = df.drop_duplicates(subset="uid")
        df = df.sort_values("uid")
        # it's a possible feature to drop uid if it's not in write_cols
        # df = df.drop(columns=["uid"])
        return df

    def _save_epoch(self, trainer, stage):
        # add barrier to make sure all processes have finished writing
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            if self.filetype == "tsv":
                # merge all tsv files and write to a new tsv file
                tsv_files = glob.glob(
                    os.path.join(self.output_dir, f"{stage}_predictions_*.tsv")
                )
                df_list = [pd.read_csv(file, sep="\t") for file in tsv_files]
                merged_df = pd.concat(df_list, ignore_index=True)
                if self.remove_duplicates:
                    merged_df = self._deduplicate_uid(merged_df)
                merged_df.to_csv(
                    os.path.join(self.output_dir, f"{stage}_predictions.tsv"),
                    index=False,
                    sep="\t",
                )

                # << cleans up the intermediate files
                if self.delete_intermediate_files:
                    for file in tsv_files:
                        os.remove(file)

            elif self.filetype == "pt":
                prediction_files = glob.glob(
                    os.path.join(self.output_dir, f"{stage}_predictions_*.pt")
                )
                merged_data = {}
                tensor_cols = {}
                for i, file in enumerate(prediction_files):
                    data = torch.load(file)
                    if i == 0:
                        for col, val in data.items():
                            if val is None:
                                continue
                            tensor_cols[col] = type(val) is torch.Tensor
                            merged_data[col] = []
                    for col, val in data.items():
                        if col not in merged_data:
                            continue
                        if tensor_cols[col]:
                            merged_data[col].append(val.cpu())
                        else:
                            merged_data[col].extend(val)
                    # Cleans up the intermediate files
                    if self.delete_intermediate_files:
                        os.remove(file)
                for col, is_tensor in tensor_cols.items():
                    if is_tensor:
                        if merged_data[col][0].dim() > 1:
                            max_length_col = max([val.shape[1] for val in merged_data[col]])
                            merge_tuples = []
                            for k, val in enumerate(merged_data[col]):
                                # defining tuple for padding with nans using F.pad
                                target_shape = [0] * (2 * len(val.shape))
                                # we want to pad to the right on the 2nd dimension
                                target_shape[2] = max_length_col - val.shape[1]
                                # F.pad takes padding values in reverse w/last dimension first
                                merge_tuple = tuple(reversed(target_shape))
                                merge_tuples.append(merge_tuple)

                            merged_data[col] = [
                                F.pad(
                                    val.float(),
                                    merge_tuples[k],
                                    mode="constant",
                                    value=float("nan"),
                                )
                                for k, val in enumerate(merged_data[col])
                            ]

                        merged_data[col] = torch.cat(merged_data[col], dim=0)

                torch.save(
                    merged_data,
                    os.path.join(self.output_dir, f"{stage}_predictions.pt"),
                )
            else:
                ValueError(f"filetype {self.filetype} does not support")

    def on_test_batch_end(
        self, trainer, pl_module, predictions, batch, batch_idx, dataloader_idx=0
    ):
        self._save_batch(trainer, predictions, batch_idx, stage="test")

    def on_predict_batch_end(
        self, trainer, pl_module, predictions, batch, batch_idx, dataloader_idx=0
    ):
        self._save_batch(trainer, predictions, batch_idx, stage="predict")

    def on_predict_epoch_end(self, trainer, pl_module):
        self._save_epoch(trainer, stage="predict")


class FTScheduler(BaseFinetuning):
    """Finetuning scheduler that gradually unfreezes layers based on a schedule

    Args:
        ft_schedule_path (str): Path to a finetuning schedule that mentions which modules to unfreeze at which epoch. See tutorial for examples.
    """

    # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/utils/finetune_callback.py

    def __init__(self, ft_schedule_path: str):
        super().__init__()

        with open(ft_schedule_path, "r") as f:
            self.ft_schedule = yaml.safe_load(f)

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        all_valid_modules = []
        for name, module in pl_module.named_modules():
            if not name:
                continue
            all_valid_modules.append(module)
        self.freeze(all_valid_modules)

    def finetune_function(
        self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer
    ) -> None:
        if current_epoch == 0:
            ## NOTE: again call freeze_before_training: since configure_model() is called AFTER the freeze_before_training() is called.
            self.freeze_before_training(pl_module)

        if current_epoch in self.ft_schedule:
            current_epoch_unfreeze_regex = re.compile(
                "|".join(self.ft_schedule[current_epoch])
            )
            for name, module in pl_module.named_modules():
                if current_epoch_unfreeze_regex.match(name):
                    self.unfreeze_and_add_param_group(
                        modules=module,
                        optimizer=optimizer,
                        initial_denom_lr=1.0,
                    )
