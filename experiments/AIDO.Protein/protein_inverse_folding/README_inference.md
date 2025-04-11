# Protein Inverse Folding
Protein inverse folding represents a computational technique aimed at generating protein sequences that will fold into specific three-dimensional structures. The central challenge in protein inverse folding involves identifying sequences capable of reliably adopting the intended structure. In our research, we concentrate on designing sequences based on the known backbone structure of a protein, represented with 3D coordinates of the atoms of the backbone (without any information about what the individual amino-acids are). Specifically. we finetune the [AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.Protein-16B) model with LoRA on the [CATH 4.2](https://pubmed.ncbi.nlm.nih.gov/9309224/) benchmark dataset. We use the same train, validation, and test splits used by the previous studies, such as [LM-Design](https://arxiv.org/abs/2302.01649), and [DPLM](https://arxiv.org/abs/2402.18567). Current version of ModelGenerator contains the inference pipeline for protein inverse folding. Experimental pipeline on other datasets (both training and testing) will be included in the future.


#### Setup Docker:
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
    # run a container (NOTE: For some of the Nvidia GPUs, you may need to replace `--runtime=nvidia` with `--gpus all` in the following command)
    docker run -d --runtime=nvidia -it -v "$(pwd):/workspace" -v "$HOME/mgen_data:/mgen_data"  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" aido /bin/bash
    # find the container ID
    docker ps # this will print the running containers and their IDs
    # execute the container with ID=<container_id>
    docker exec -it <container_id> /bin/bash  # now you should be inside the docker container
    # test if you can access the nvidia GPUs
    nvidia-smi # this should print the GPUs' details
    # make sure wget and git are installed
    apt update && apt install -y wget git
    ```
- Execute the following steps from **within** the docker container you just created.
- **Note:** Multi-GPU inference for inverse folding is not currently supported and will be included in the future.


#### Download and merge model checkpoint chunks:

- From the terminal, change directory to "/workspace/experiments/AIDO.Protein/protein_inverse_folding/":
  ```
  cd /workspace/experiments/AIDO.Protein/protein_inverse_folding
  ```

- Download all the 15 model checkpoint chunks (named as `chunk_<chunk_ID>.bin`) from [here](https://huggingface.co/genbio-ai/AIDO.ProteinIF-16B/tree/main). Place them inside the directory `${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model_chunks` and merge them.

  You can do this by simply running the following script:
  ```
  mkdir -p ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/
  huggingface-cli download genbio-ai/AIDO.ProteinIF-16B \
  --repo-type model \
  --local-dir ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/
  # Merge chunks
  python merge_ckpt.py ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model_chunks ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model.ckpt
  ```

#### Setup necessary variables:
- Set necessary variables in terminal. We are using PDB entry 5YH2 (chain A) as an example here:
  ```
  DATA_DIR=data ## Set the path of the directory where you want to keep/download our PDB/CIF file.
  PDB_ID=5YH2
  CHAIN_ID=A
  ```


#### Download and preprocess PDB/CIF files of protein 3D structures:
- Download a single structure from somewhere like PDB:
  ```
  mkdir -p ${DATA_DIR}/
  wget https://files.rcsb.org/download/${PDB_ID}.cif -P ${DATA_DIR}/
  ```

- Put it into our formats:
  ```
  python preprocess_PDB.py ${DATA_DIR}/${PDB_ID}.cif ${CHAIN_ID} ${DATA_DIR}/
  ```


#### Run inference:
- Run inference to generate the sequence (evaluation scores will also be computed based on the native sequence in the input PDB/CIF file):
  ```
  mgen test \
    --config protein_inv_fold_test.yaml \
    --trainer.default_root_dir ${MGEN_DATA_DIR}/modelgenerator/logs/protein_inv_fold/ \
    --ckpt_path ${MGEN_DATA_DIR}/modelgenerator/huggingface_models/protein_inv_fold/AIDO.ProteinIF-16B/model.ckpt \
    --trainer.devices 0, \
    --data.path ${DATA_DIR}/
  ```


#### Outputs:
- The evaluation scores will be printed on the console (if the native sequence in the input PDB/CIF file is a dummy, please ignore the evaluation scores). For the aforementioned example (PDB entry: 5YH2, chain A), the following would be printed in a table:
    ```
    test/acc            0.7102803587913513
    test/loss           1.4816021919250488
    ```
- The outputs will be saved in the file **`/workspace/experiments/AIDO.Protein/protein_inverse_folding/proteinIF_outputs/results_acc_<recovery_accuracy>.txt`**:
  - Here, we have three lines of information:
    - Line1: Identity of the protein (as '`name=<PDB_ID>.<CHAIN_ID>`'), length of the squence (as '`L=<length_of_sequence>`'), and the recovery rate/accuracy for that protein sequence (as '`Recovery=<recovery_rate_of_sequence>`')
    - Line2: *Single-letter representation* of amino-acids of the ground truth sequence (as `true:<sequence_of_amino_acids>`)
    - Line3: *Single-letter representation* of amino-acids of the predicted sequence by our method (as `pred:<sequence_of_amino_acids>`)
  - For the aforementioned example (PDB entry: 5YH2, chain A), the output file has the following content (the file is named "results_acc_0.7102803587913513.txt"):
      ```
      >name=5YH2.A | L=428 | Recovery=0.7102803587913513
      true:SSKLQALFAHPLYNVPEEPPLLGAEDSLLASQEALRYYRRKVARWNRRHKMDPPLQLRLEASWVQFHLGINRHGLYSRSSPVVSKLLQDMRHFPTISADYSQDEKALLGACDCTQIVKPSGVHLKLVLRFSDFGKAMFKPMRQQRDEETPVDFFYFIDFQRHNAEIAAFHLDRILDFRRVPPTVGRIVNVTKEILEVTKNEILQSVFFVSPASNVCFFAKCPYMCKTEYAVCGKPHLLEGSLSAFLPSLNLAPRLSVPNPWIRSYTLAGKEEWEVNPLYCDTVKQIYPYNNSQRLLNVIDMAIFDFLIGNMDRHHYEMFTKFGDDGFLIHLDNARGFGRHSHDEISILSPLSQCCMIKKKTLLHLQLLAQADYRLSDVMRESLLEDQLSPVLTEPHLLALDRRLQTILRTVEGCIVAHGQQSVIVDGP
      pred:MSPLEKLFNHPLYNIPVLPLLLGEDTILLDKEKALKYYKKLTKKFNLPLKKKPPLVFKEDASWVQFHLGITRHGVYSRSSPVVSKLLQDMRTLPVISVDGGGTLKALKGACDCSQLQKPSGTQLKLLVKFQNFGKALFKPMRQQRDEETPEDFFYYSDYERHNAEIAAFHLDRILDFRRVPPTVGRLVNVTKELYDVTKDNKLRSTFFISPDNNVCFFAKCPYYCDTTHVVCGNPDLLEGSLAAFLPDKNLAPRKSIPSPWIRSYTLSGKEEWEVDPDYCDTVKQIYPYNSSNRLLNIIDMSIFDFLIGNMDRHHYETFTKFGDDGFLIHLDNAKGFGRHSHDELSILAPLTQCCMIRRSTLLRLQLLSKEEVRLSDVLRESLLEDSLYPVLTEPHLLAFDRRLQIILKTVEGCLKKKGEKETIYDGP
      ```

