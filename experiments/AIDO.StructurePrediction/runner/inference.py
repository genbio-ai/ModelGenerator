
import logging
import os, sys
import traceback
from contextlib import nullcontext
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, Mapping
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

import argparse
import yaml
import ml_collections
import torch
import torch.distributed as dist
from runner.dumper import DataDumper
from runner.dump_manager import lock_manager, pid_issame

from fold.data.ccd_data import set_components_file, set_rkdit_mol_pkl
from fold.data.infer_data_pipeline import get_inference_dataloader
from fold.model.fold import Fold
from fold.utils.distributed import DIST_WRAPPER
from fold.utils.seed import seed_everything
from fold.utils.torch_utils import to_device

from fold.utils.logger import Logger

logger = Logger.logger



class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(need_atom_confidence=configs.need_atom_confidence)

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set `CUTLASS_PATH` env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        self.model = Fold(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = self.configs.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=True,
        )
        self.model.eval()
        self.print(f"Finish loading checkpoint.")

    def init_dumper(self, need_atom_confidence: bool = False):
        self.dumper = DataDumper(
            base_dir=self.dump_dir, need_atom_confidence=need_atom_confidence
        )

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        return prediction

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def update_inference_configs(configs: Any, N_token: int):
    # Setting the default inference configs for different N_token and N_atom
    # when N_token is larger than 3000, the default config might OOM even on a
    # A100 80G GPUS,
    if N_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif N_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    return configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    # Data
    logger.info(f"Loading data from {configs.input_json_path}")
    dataloader = get_inference_dataloader(configs=configs)
    dataset_name = ""
    num_data = len(dataloader.dataset)
    for seed in configs.seeds.split(","):
        seed = int(seed)
        seed_everything(seed=seed, deterministic=True)
        for batch in dataloader:
            try:
                data, atom_array, data_error_message = batch[0]
                if len(data_error_message) > 0:
                    logger.info(data_error_message)
                    with open(
                        opjoin(runner.error_dir, f"{data['sample_name']}.txt"),
                        "w",
                    ) as f:
                        f.write(data_error_message)
                    continue

                sample_name = data["sample_name"]

                logger.info("--------------------------")
                # #################################
                dump_dir = runner.dumper._get_dump_dir(dataset_name, f"{sample_name}", seed)
                logger.info(f"dump_dir={dump_dir}")
                os.system(f"mkdir -p '{dump_dir}'")
                with lock_manager(dump_dir) as pid:
                    if pid_issame(pid, str(os.getpid()).strip()):
                        logger.info(
                            (
                                f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                                f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                                f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                            )
                        )
                        new_configs = update_inference_configs(configs, data["N_token"].item())
                        runner.update_model_configs(new_configs)
                        
                        prediction = runner.predict(data)
                        runner.dumper.dump(
                            dataset_name=dataset_name,
                            pdb_id=sample_name,
                            seed=seed,
                            pred_dict=prediction,
                            atom_array=atom_array,
                            entity_poly_type=data["entity_poly_type"],
                        )
                        logger.info(
                            f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded.\n"
                            f"Results saved to {configs.dump_dir}"
                        )
                        torch.cuda.empty_cache()
                        ###
                    else:
                        logger.info(f"skip {sample_name}... ")
                ##################################
                
            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                # Save error info
                if opexists(
                    error_path := opjoin(runner.error_dir, f"{sample_name}.txt")
                ):
                    os.remove(error_path)
                with open(error_path, "w") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
                raise RuntimeError(f"run infer failed: {str(e)}")


def main(configs: Any) -> None:
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)

def load_yaml(yaml_path):
    """
    load yaml and convert it to ml_collection
    :param yaml_path: filename to save
    :return:
        config: ml_collections
    """
    with open(yaml_path, "r") as f:
        instance = yaml.load(f, Loader=yaml.SafeLoader)
    config = ml_collections.ConfigDict(instance)

    return config

def convert_config_to_dict(config):
    if isinstance(config, ml_collections.ConfigDict):
        return {key: convert_config_to_dict(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [convert_config_to_dict(item) for item in config]
    else:
        return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_file_path",
        type=str,
        help="yaml_file_path",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="load checkpoint",
    )
    parser.add_argument(
        "--ccd_components_file",
        type=str,
        help="ccd_components_file",
    )
    parser.add_argument(
        "--ccd_components_rdkit_mol_file",
        type=str,
        help="ccd_components_rdkit_mol_file",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="a set of seeds",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        help="dump directory",
    )
    parser.add_argument(
        "--input_json_path",
        type=str,
        help="input_json_path",
    )

    args = parser.parse_args()

    return args


def run(args):
    configs = load_yaml(args.yaml_file_path)
    configs.checkpoint_path = args.checkpoint_path
    configs.seeds = args.seeds
    configs.dump_dir = args.dump_dir
    configs.input_json_path = args.input_json_path
    set_components_file(args.ccd_components_file)
    set_rkdit_mol_pkl(args.ccd_components_rdkit_mol_file)
    main(configs)

if __name__ == "__main__":
    args = get_args()
    run(args)
    logger.info("end.")
