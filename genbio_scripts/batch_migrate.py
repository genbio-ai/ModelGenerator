import glob
import torch
import yaml
from pathlib import Path
from huggingface_hub import list_models, snapshot_download, upload_folder
import warnings
warnings.filterwarnings("ignore", ".*resume_download*")

all_models = [model.id for model in list_models(search="genbio-ai")]
match_exclude = (
    # backbones
    "genbio-ai/AIDO.DNA-7B",
    "genbio-ai/AIDO.DNA-300M",
    "genbio-ai/AIDO.DNA-dummy",
    "genbio-ai/AIDO.RNA-1.6B",
    "genbio-ai/AIDO.RNA-1.6B-CDS",
    "genbio-ai/AIDO.RNAIF-1.6B",
    "genbio-ai/AIDO.RNA-650M",
    "genbio-ai/AIDO.RNA-650M-CDS",
    "genbio-ai/AIDO.RNA-300M-MARS",
    "genbio-ai/AIDO.RNA-25M-MARS",
    "genbio-ai/AIDO.RNA-1M-MARS",
    "genbio-ai/AIDO.Protein-16B",
    "genbio-ai/AIDO.Protein-16B-v1",
    "genbio-ai/AIDO.ProteinIF-16B",
    "genbio-ai/AIDO.Protein2StructureToken-16B",
    # invalid repo structure
    "genbio-ai/AIDO.RNA-1.6B-bpRNA_secondary_structure_prediction",
    # excluded due to size
    'genbio-ai/AIDO.RNA-1.6B-translation-efficiency-muscle',
    'genbio-ai/AIDO.RNA-1.6B-mrna-expression-level-pc3',
)
prefix_exclude = (
    "genbio-ai/AIDO.Structure",
    "genbio-ai/AIDO.Cell"
)
models_to_migrate = list(filter(lambda model: model not in match_exclude and not model.startswith(prefix_exclude), all_models))
print(models_to_migrate, len(models_to_migrate))

for model_id in models_to_migrate:
    download_path = str(Path("~/genbio_models").expanduser().joinpath(model_id))
    print(f">> Downloading {model_id}")
    download_path = snapshot_download(model_id, local_dir=download_path)
    conf_paths = glob.glob(f"{download_path}/**/config.yaml", recursive=True)
    ckpt_paths = glob.glob(f"{download_path}/**/model.ckpt", recursive=True)
    if not conf_paths or not ckpt_paths:
        print(f"Skipping empty repo: {model_id}")
        continue
    if len(conf_paths) == 1:
        conf_paths = conf_paths * len(ckpt_paths)
    assert len(conf_paths) == len(ckpt_paths)
    for conf_path, ckpt_path in zip(conf_paths, ckpt_paths):
        print(ckpt_path, conf_path)
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        with open(conf_path, "rt") as f:
            config = yaml.safe_load(f)
        # proteinfm_ct -> aido_protein_16b_v1
        if "proteinfm_ct" in config["model"]["init_args"]["backbone"]["class_path"]:
            config["model"]["init_args"]["backbone"]["class_path"] = config["model"]["init_args"]["backbone"]["class_path"].replace("proteinfm_ct", "aido_protein_16b_v1")
            with open(conf_path, "wt") as f:
                yaml.safe_dump(config, f, sort_keys=False)
        # how the hack could this even happen??
        checkpoint["hyper_parameters"].pop("n_labels", None)
        checkpoint["hyper_parameters"].pop("n_classes", None)

        checkpoint["hyper_parameters"].update(config["model"]["init_args"])
        checkpoint["hyper_parameters"]["_class_path"] = config["model"]["class_path"]
        with open(ckpt_path, "wb") as f:
            torch.save(checkpoint, f)
    upload_folder(repo_id=model_id, folder_path=download_path, commit_message="migrate checkpoint based on config.yaml")
