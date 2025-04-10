# RNA Inverse Folding
RNA inverse folding is a computational method designed to create RNA sequences that fold into predetermined three-dimensional structures. Our study focuses on generating sequences using the known backbone structure of an RNA, defined by the 3D coordinates of its backbone atoms, without any information of the individual bases. Specifically. we fully finetune the [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) model on the single-state split from [Das _et al._](https://www.nature.com/articles/nmeth.1433) already processed by [Joshi _et al._](https://arxiv.org/abs/2305.14749). We use the same train, validation, and test splits used by their method [gRNAde](https://arxiv.org/abs/2305.14749). Current version of ModelGenerator contains the inference pipeline for RNA inverse folding. Experimental pipeline on other datasets (both training and testing) will be included in the future.

##### Experimental Results
We evaluate our model in two settings: (1) adaptation with conditional diffusion where AIDO.RNA is fine-tuned for the inverse folding task; and (2) zero-shot generation where AIDO.RNA is frozen.
| Model | Mean Sequence Recovery Rate |
| ------ | ------ |
|    gRNAde    |    52.78    |
|    gRNAde+AIDO.RNA-Zeroshot    |    53.16    |
|    gRNAde+AIDO.RNA-Finetuned.  |    54.41    |

##### Acknowledgement 
We thank the authors of [gRNAde](https://arxiv.org/abs/2305.14749) for providing their models' checkpoints and configuration files.

#

In the following sections, we discuss how to use AIDO.RNA for RNA inverse folding using ModelGenerator.

#### Setup
Install [ModelGenerator](https://github.com/genbio-ai/modelgenerator). 
- It is **required** to use [docker](https://www.docker.com/101-tutorial/) to run our inverse folding pipeline.
- Please set up a docker image using our provided [Dockerfile](https://github.com/genbio-ai/ModelGenerator/blob/main/Dockerfile) and run the inverse folding inference from within the docker container. 

Here is an example bash script to set up and access a docker container:
```
# clone the ModelGenerator repository
git clone https://github.com/genbio-ai/ModelGenerator.git
# cd to "ModelGenerator" folder where you should find the "Dockerfile"
cd ModelGenerator
# create a docker image
docker build -t aido .
# create a local folder as ModelGenerator's data directory
mkdir -p $HOME/mgen_data
# run a container
docker run -d --runtime=nvidia -it -v "$(pwd):/workspace" -v "$HOME/mgen_data:/mgen_data" aido /bin/bash
# find the container ID
docker ps # this will print the running containers and their IDs
# execute the container with ID=<container_id>
docker exec -it <container_id> /bin/bash  # now you should be inside the docker container
# test if you can access the nvidia GPUs
nvidia-smi # this should print the GPUs' details
```
- Execute the following steps from **within** the docker container you just created.
- **Note:** Multi-GPU inference for inverse folding is not currently supported and will be included in the future.

#### Download model checkpoints

- Download the `model.ckpt` checkpoint from [here](https://huggingface.co/genbio-ai/AIDO.RNAIF-1.6B/blob/main/model.ckpt). Place it inside the local directory `${MGEN_DATA_DIR}/modelgenerator/huggingface_models/rna_inv_fold/AIDO.RNAIF-1.6B`.

- Download the gRNAde checkpoint named `gRNAde_ARv1_1state_das.h5` from the [huggingface-hub](https://huggingface.co/genbio-ai/AIDO.RNAIF-1.6B/blob/main/other_models/gRNAde_ARv1_1state_all.h5) ***or*** the [original source](https://github.com/chaitjo/geometric-rna-design/blob/main/checkpoints/gRNAde_ARv1_1state_all.h5). Place it inside the directory `${MGEN_DATA_DIR}/modelgenerator/huggingface_models/rna_inv_fold/AIDO.RNAIF-1.6B/other_models`. Set the environment variable `gRNAde_CKPT_PATH=${MGEN_DATA_DIR}/modelgenerator/huggingface_models/rna_inv_fold/AIDO.RNAIF-1.6B/other_models/gRNAde_ARv1_1state_das.h5`

**Alternatively**, you can simply run the following script to do both of these steps:
```
mkdir -p ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/rna_inv_fold/AIDO.RNAIF-1.6B
huggingface-cli download genbio-ai/AIDO.RNAIF-1.6B \
--repo-type model \
--local-dir ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/rna_inv_fold/AIDO.RNAIF-1.6B
# Set the environment variable gRNAde_CKPT_PATH
export gRNAde_CKPT_PATH=${MGEN_DATA_DIR}/modelgenerator/huggingface_models/rna_inv_fold/AIDO.RNAIF-1.6B/other_models/gRNAde_ARv1_1state_das.h5
```

#### Download data
Download the data preprocessed by [Joshi _et al._](https://arxiv.org/abs/2305.14749). Mainly download these two files: [processed.pt.zip](https://huggingface.co/datasets/genbio-ai/rna-inverse-folding/blob/main/processed.pt.zip) and [processed_df.csv](https://huggingface.co/datasets/genbio-ai/rna-inverse-folding/blob/main/processed_df.csv). Place them inside the directory `${MGEN_DATA_DIR}/modelgenerator/datasets/rna_inv_fold/raw_data/`. Please refer to [this link](https://github.com/chaitjo/geometric-rna-design/tree/main?tab=readme-ov-file#downloading-and-preparing-data) for details about the dataset and its preprocessing.

**Alternatively**, you run the following script to do it:
```
mkdir -p ${MGEN_DATA_DIR}/modelgenerator/datasets/rna_inv_fold/raw_data/
huggingface-cli download genbio-ai/rna-inverse-folding \
--repo-type dataset \
--local-dir ${MGEN_DATA_DIR}/modelgenerator/datasets/rna_inv_fold/raw_data/
```

#### Run inference
From your terminal, change directory to `experiments/AIDO.RNA/rna_inverse_folding` folder and run the following script:
```
cd modelgenerator/rna_inv_fold/gRNAde_structure_encoder
echo "Running inference.."
python main.py
echo "Extracting structure encoding.."
python main_encoder_only.py
cd  ../../../experiments/AIDO.RNA/rna_inverse_folding/
# run inference
mgen test --config rna_inv_fold_test.yaml \
  --trainer.default_root_dir ${MGEN_DATA_DIR}/modelgenerator/logs/rna_inv_fold/ \
  --ckpt_path ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/rna_inv_fold/AIDO.RNAIF-1.6B/model.ckpt \
  --trainer.devices 0, \
  --data.path ${MGEN_DATA_DIR}/modelgenerator/datasets/rna_inv_fold/structure_encoding/
```

#### Outputs
- The evaluation score will be printed on the console.
- The generated sequences will be stored in `./rnaIF_outputs/designed_sequences.json`. In this file, we will have:
  1. **`"true_seq"`**: the ground truth sequences,
  2. **`"pred_seq"`**: predicted sequences by our method,
  3. **`"baseline_seq"`**: predicted sequences by the baseline method [gRNAde](https://arxiv.org/abs/2305.14749).

An example file content with two test samples is shown below:
```
{
  "true_seq": [
    "CCCAGUCCACCGGGUGAGAAGGGGGCAGAGAAACACACGACGUGGUGCAUUACCUGCC",
    "UCCCGUCCACCGCGGUGAGAAGGGGGCAGAGAAACACACGAUCGUGGUACAUUACCUGCC",
  ],
  "pred_seq": [
    "UGGGGAGCCCCCGGGGUGAACCAGCCGGUGAAAGGCACCCGGUGAUCGGUCAGCCCAC",
    "GCGGAUGCCCCGCCCGGUCAACCGCAUGGUGAAAUCCACGCGCCUGGUGGGUUAGCCAUG",
  ],
  "baseline_seq": [
    "UGGUGAGCCCCCGGGGUGAACCAGUAGGUGAAAGGCACCCGGUGAUCGGUCAGCCCAC",
    "GCGGAUGCCGGGCCCGGUCCACCGCAUGGUGAAAUUCAGGCGCCUGGAGGGUUAGCCAUG",
  ]
}
```