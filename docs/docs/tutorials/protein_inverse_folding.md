# Protein Inverse Folding
Protein inverse folding represents a computational technique aimed at generating protein sequences that will fold into specific three-dimensional structures. The central challenge in protein inverse folding involves identifying sequences capable of reliably adopting the intended structure. In our research, we concentrate on designing sequences based on the known backbone structure of a protein, represented with 3D coordinates of the atoms of the backbone (without any information about what the individual amino-acids are). Specifically. we finetune the [AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.Protein-16B) model with LoRA on the [CATH 4.2](https://pubmed.ncbi.nlm.nih.gov/9309224/) benchmark dataset. We use the same train, validation, and test splits used by the previous studies, such as [LM-Design](https://arxiv.org/abs/2302.01649), and [DPLM](https://arxiv.org/abs/2402.18567). Current version of ModelGenerator contains the inference pipeline for protein inverse folding. Experimental pipeline on other datasets (both training and testing) will be included in the future.

#### Setup:
Install [ModelGenerator](https://github.com/genbio-ai/modelgenerator). 
- It is **required** to use [docker](https://www.docker.com/101-tutorial/) to run our inverse folding pipeline.
- Please set up a docker image using our provided [Dockerfile](https://github.com/genbio-ai/ModelGenerator/blob/main/Dockerfile) and run the inverse folding inference from within the docker container. 
  - Here is an example bash script to set up and access a docker container:
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

#### Download and merge model checkpoint chunks:

- Download all the 15 model checkpoint chunks (named as `chunk_<chunk_ID>.bin`) from [here](https://huggingface.co/genbio-ai/AIDO.ProteinIF-16B/tree/main). Place them inside the directory `${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model_chunks`. 

  **Alternatively**, you can do this by simply running the following script:
  ```
  mkdir -p ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/
  huggingface-cli download genbio-ai/AIDO.ProteinIF-16B \
  --repo-type model \
  --local-dir ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/
  # Merge chunks
  python merge_ckpt.py ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model_chunks ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model.ckpt
  ```

#### Download data:
- Download the preprocessed CATH 4.2 dataset from [here](https://huggingface.co/datasets/genbio-ai/protein-inverse-folding/tree/main/cath-4.2).  You should find two files named [chain_set_map.pkl](https://huggingface.co/datasets/genbio-ai/protein-inverse-folding/blob/main/cath-4.2/chain_set_map.pkl) and [chain_set_splits.json](https://huggingface.co/datasets/genbio-ai/protein-inverse-folding/blob/main/cath-4.2/chain_set_splits.json). Place them inside the directory `${MGEN_DATA_DIR}/modelgenerator/datasets/protein_inv_fold/cath_4.2/`. (Note that it was originally preprocessed by [Generative Models for Graph-Based Protein Design (Ingraham et al, NeurIPS'19)](https://papers.nips.cc/paper_files/paper/2019/file/f3a4ff4839c56a5f460c88cce3666a2b-Paper.pdf), and we further preprocessed it to suit our pipeline.)
  
  **Alternatively**, you can do it by simply running the following script:
  ```
  mkdir -p ${MGEN_DATA_DIR}/modelgenerator/datasets/protein_inv_fold/cath_4.2/
  huggingface-cli download genbio-ai/protein-inverse-folding \
  --repo-type dataset \
  --local-dir ${MGEN_DATA_DIR}/modelgenerator/datasets/protein_inv_fold
  ```

#### Run inference:
- From your terminal, change directory to `experiments/AIDO.Protein/protein_inverse_folding` folder and run the following script:
    ```
    cd experiments/AIDO.Protein/protein_inverse_folding
    # Run inference
    mgen test --config protein_inv_fold_test.yaml \
            --trainer.default_root_dir ${MGEN_DATA_DIR}/modelgenerator/logs/protein_inv_fold/ \
            --ckpt_path ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model.ckpt \
            --trainer.devices 0, \
            --data.path ${MGEN_DATA_DIR}/modelgenerator/datasets/protein_inv_fold/cath_4.2/
    ```

#### Outputs:
- The evaluation score will be printed on the console. 
- The generated sequences will be stored the folder `proteinIF_outputs/`. There will be two output files:
  - **`./proteinIF_outputs/designed_sequences.pkl`**: This file will contain the raw token (amino-acid) IDs of the ground truth sequences (`"true_seq"`) and predicted sequences by our method (`"pred_seq"`), stored as numpy arrays. An example:
    ```
    {
      'true_seq': [
          array([[ 4,  8,  4,  3, 12,  5,  2, 11, 16, 15,  5,  1, 11, ...]]), ...
      ],
      'pred_seq': [
          array([[ 8,  2,  4,  3, 10,  6,  2, 11, 16, 15,  6,  1, 11, ...]]), ...
      ]
    }
    ```
  - **`./proteinIF_outputs/results_acc_<median_accuracy>.txt`** (where median accuracy is the median accuracy calculated over all the test samples):
    - Here, for each protein in the test set, we have three lines of information:
      - Line1: Identity of the protein (as '`name=<PDB_ID>.<CHAIN_ID>`'), length of the squence (as '`L=<length_of_sequence>`'), and the recovery rate/accuracy for that protein sequence (as '`Recovery=<recovery_rate_of_sequence>`')
      - Line2: *Single-letter representation* of amino-acids of the ground truth sequences (as `true:<sequence_of_amino_acids>`)
      - Line3: *Single-letter representation* of amino-acids of the predicted sequences by our method (as `pred:<sequence_of_amino_acids>`)
    - An example file content:
      ```
      >name=3fkf.A | L=141 | Recovery=0.5957446694374084
      true:VTVGKSAPYFSLPNEKGEKLSRSAERFRNRYLLLNFWASWCDPQPEANAELKRLNKEYKKNKNFAMLGISLDIDREAWETAIKKDTLSWDQVCDFTGLSSETAKQYAILTLPTNILLSPTGKILARDIQGEALTGKLKELL
      pred:TAVGDEAPYFELPDLEGKKLSLDSEEFKNKYLLLDFWASWCLPCREEIAELKELYRRFAKNKKFAILGVSADTDKEAWLKAVKEDNLRWTQVSDFKGWDSEVFKNYNVQSLPENILLSPEGKILARGIRGEALRNKLKELL
      
      >name=2d9e.A | L=121 | Recovery=0.7685950398445129
      true:GSSGSSGFLILLRKTLEQLQEKDTGNIFSEPVPLSEVPDYLDHIKKPMDFFTMKQNLEAYRYLNFDDFEEDFNLIVSNCLKYNAKDTIFYRAAVRLREQGGAVLRQARRQAEKMGSGPSSG
      pred:GSSGSSGRLTLLRETLEQLQERDTGWVFSEPVPLSEVPDYLDVIDHPMDFSTMRRKLEAHRYLSFDEFERDFNLIVENCRKYNAKDTVFYRAAVRLQAQGGAILRKARRDVESLGSGPSSG
      ```