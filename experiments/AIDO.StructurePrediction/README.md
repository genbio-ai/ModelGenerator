# [AIDO](https://github.com/genbio-ai/aido).StructurePrediction

| Antibody | Nanobody | RNA |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="https://cdn-uploads.huggingface.co/production/uploads/67f69b92e9fa2b2fdb84053d/HRkmsQGorXEpxQtauXBNe.gif" width="300"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/67f69b92e9fa2b2fdb84053d/TiuDROAYDIEp4ac-4OzXB.gif" width="300"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/67f69b92e9fa2b2fdb84053d/8gD1JOiDywkhVYLlsmngM.gif" width="300">|


| Our Model | AlphaFold3 |
|:-------------------------:|:-------------------------:|
|<img src="assets/img/figure(gt-yellow vs our-green).png" height="400"> | <img src="assets/img/figure(gt-yellow vs af3-blue).png" height="400">|

## Model Description

AIDO.StructurePrediction is an AlphaFold3-like full-atom structure prediction model,
designed to predict the structure and interactions of biological molecules,
including proteins, DNA, RNA, ligands, and antibodies. This model harnesses both structural and sequence modalities
to provide high-fidelity predictions for various biological tasks.
Our model achieved state-of-the-art performance on immunize-related structure prediction tasks,
including antibody, nanobody, antibody-antigen, and nanobody-antigen.

## Model Details

### Key Features

- **Multi-Modal Learning**: Combines 3D structural and sequence data (nucleotides and amino acids) to enhance model accuracy and applicability.
- **High-Quality Data**: We have used carefully curated structure data when training the model.
- **Data Augmentation**: Implements novel data augmentation and distillation techniques to diversify training datasets, improving robustness and generalization.
- **Integration of Multiple Sequence Alignments (MSA)**: Utilizes alignment data from diverse biological databases to improve predictive capabilities.
- **Training Strategies**: Incorporates advanced training methodologies to refine model performance and efficiency.

### Model Architecture

- **Type**: Pairiformer and Diffusion Module.
- **Key Components**:
  - Pairformer for learning complex relationships from single sequences and from multiple sequence alignments.
  - Diffusion module to generate multiple structure conformations.

| Model Arch Component    | Value |
|-------------------------|:-----:|
| Pairformer Blocks       |  48   |
| MSA Module Blocks       |   4   |
| Diffusion Module Blocks |  24   |
| Diffusion Heads         |  16   |

## Quickstart

AIDO.StructurePrediction comes with a CLI tool `genbio-aidosp` to download model checkpoint and prepare the necessary data for inference.

If you have already prepared all the necessary inputs, you can skip to the [Inference section](#inference)

The CLI utilities can be installed with `pip` or `uv`:

```bash
git clone https://github.com/genbio-ai/ModelGenerator
cd ModelGenerator/experiments/AIDO.StructurePrediction

# We suggest to create a separate virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install . # or uv pip install .
 ```

### Example Run
```bash
# Download model and CCD component files
genbio-aidosp util download-model -o my_model_dir --model-version v0.1.2
genbio-aidosp util download-ccd -o my_ccd_dir

# Create your job JSON
# examples/example.json

# Create a local MSA database for your job
python job2fasta.py --input examples/example.json --output examples/example.fasta
genbio-aidosp util retrieve-msa -i examples/example.fasta -o my_msa_dir -c configs/mmseqs.yaml --cpus-per-task 8 --no-tasks 1 --shuffle-file-list

# Build the full job JSON with MSAs
python buildjob.py --input examples/example.json --msa-db my_msa_dir --output examples/example_built.json

# Build the image
docker build -t fold_pyt -f Dockerfile .
export CHECKPOINT_DIR=${PWD}/my_model_dir
export CCD_DIR=${PWD}/my_ccd_dir
export LOCAL_MSA_DATABASE=${PWD}/my_msa_dir
docker run --network=host --rm --gpus all -it \
  --name aido-structure-prediction \
  --shm-size=1000g \
  -v $CHECKPOINT_DIR:/nfs/model \
  -v $CCD_DIR:/nfs/ccd \
  -v $LOCAL_MSA_DATABASE:/msa_database \
  -v $(pwd):/workspace \
  fold_pyt /bin/bash

# Run inference from within the container
cd scripts
bash run_inference.sh

# Exit the container and view the results
# visualize.ipynb
```

## Usage

### Download Model Checkpoint

You can download model checkpoint and CCD component files from [GenBio Huggingface](https://huggingface.co/genbio-ai/AIDO.StructurePrediction/tree/main) or use the packaged CLI.

```bash
genbio-aidosp util download-model -o <model_download_dir> --model-version v0.1.2 # The only available version for now
genbio-aidosp util download-ccd -o <ccd_download_dir>
```

### Retrieve MSA

Before running inference, you need to prepare the local MSA database for your target sequences. 
Compile all of the protein sequences you want to predict on into a single fasta file,
then run the following command:

```bash
genbio-aidosp util retrieve-msa -i <input_fasta_path> -o <msa_output_dir> -c configs/mmseqs.yaml --cpus-per-task <num_cpus> --no-tasks <num_tasks> --shuffle-file-list
```

> [!warning]
> For ease-of-use, this tool has been recently adapted to use the mmseq2 API and is still being tested.
> If you run into issues, please default to creating a [local MSA database](#local-msa-database).

### Prepare input JSON file

We strongly recommend that users get started with the provided [default example](examples/example.json) for their first run. 

> [!warning]
> Currently, we do not provide a tool to generate the input JSON file.
> To manage MSAs for multimers, we recommend creating a separate MSA database for each fasta.

## Inference

Assuming you have downloaded the checkpoint, and set up the local MSA database for prediction, you can use the provided docker image to start the inference container:

```bash
# Build the image
docker build -t fold_pyt -f Dockerfile .

# Run the container from the root of AIDO.StructurePrediction
export CHECKPOINT_DIR=<model_download_dir>
export CCD_DIR=<ccd_download_dir>
export LOCAL_MSA_DATABASE=<msa_output_dir>
docker run --network=host --rm --gpus all -it \
  --name aido-structure-prediction \
  --shm-size=1000g \
  -v $CHECKPOINT_DIR:/nfs/model \
  -v $CCD_DIR:/nfs/ccd \
  -v $LOCAL_MSA_DATABASE:/msa_database \
  -v $(pwd):/workspace \
  fold_pyt /bin/bash
```

Once in the container interactive shell, inference can be run with the bash script:

```bash
cd scripts
bash run_inference.sh
```

Arguments can be set inside the script and are explained as follows:

- `input_json_path`: path to a JSON file that fully describes the input.
- `checkpoint_path`: the model checkpoint path, you can download it in [Huggingface](https://huggingface.co/genbio-ai/AIDO.StructurePrediction/tree/main)
- `ccd_components_file`: ccd file. you can download it in [Huggingface](https://huggingface.co/genbio-ai/AIDO.StructurePrediction/tree/main)
- `ccd_components_rdkit_mol_file`: ccd file. you can download it in [Huggingface](https://huggingface.co/genbio-ai/AIDO.StructurePrediction/tree/main)
- `seed`: random seed, integer
- `input_json_path`: input path for json file, you can see how to make this file in data/example.json
- `output_dir`: path to a directory where the results of the inference will be saved.

### Visualize the results

Finally, you can visualize the results using the provided notebook [`visualize.ipynb`](visualize.ipynb).

<img src="assets/img/vis.png" alt="AIDO.StructurePrediction outputs"/>

## Local MSA Database

AIDO.StructurePrediction uses the MMseqs2 API to retrieve MSA data. 
This is a great option for one-off predictions, but it can be slow and unreliable for benchmarking or frequent use.

If you plan to use AIDO.StructurePrediction regularly, we recommend downloading and creating your MSA databses locally.

> [!caution]
> The unarchived databases will take up ~600Gb.

Install `aria2` from your preferred package manager. For example, on Ubuntu, you can install it using:

```bash
sudo apt-get install aria2
```

If you would like to download the expandable profile databases instead of using the MMseqs2 API, you can use the CLI:

```bash
genbio-aidosp util download-db -o <output_dir> -n colabfold_envdb -n uniref30
```

This effectively runs bash scripts in the [scripts](src/genbio/aidosp/scripts) directory, downloading and using `tar` to unpack the databases. You can also download the databases manually.

```bash
bash src/genbio/aidosp/scripts/download_colabfold_envdb.sh <output_dir>
bash src/genbio/aidosp/scripts/download_uniref30.sh <output_dir>
```

Now you can go grab a coffee while ~200Gb of databases are being downloaded ☕

### Prepare local MSA database

1. Install `mmseqs2` in your environment. For directions, see the [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2).
2. Create an index for the databases.

```bash
cd <output_dir> # From previous step
cd colabfold_envdb_202108
mmseqs tsv2exprofiledb colabfold_envdb_202108 colabfold_envdb_202108_db
mmseqs createindex colabfold_envdb_202108_db tmp

cd ../
cd uniref30_2103
mmseqs tsv2exprofiledb uniref30_2103 uniref30_2103_db
mmseqs createindex uniref30_2103_db tmp1
```

You may now use this database by modifying `configs/mmseqs.yaml`
- Set `tools.mmseqs2_api.enable` to `false`
- Set `tools.mmseqs2.enable` to `true`

After this, you can run the `retrieve-msa` command again to create the MSA database, and update your input (e.g. `examples/example.json`) to point to the new database entries.


## **Model Performance**

### Model Evaluation Metrics

**RMSD**: Root Mean Square Deviation between prediction and ground truth.

- **Protein/Antibody**: We calculate the RMSD for Cα atoms.
- **DNA/RNA**: We calculate the RMSD for C1 atoms.
- **Ligand**: We use the coordinates of all atoms.

When calculating RMSD for protein-ligand, RNA-ligand, and DNA-ligand interactions, if we use only Cα and C1 for
proteins, RNAs, and DNAs, while using full atom coordinates for ligands, the metric may be affected
by the number of atoms in the ligand. This could create potential issues. We plan to address this problem in the future.

**DockQ**:
We modified the script based on [this public repo](https://github.com/bjornwallner/DockQ) to support missing residues.

**Note**: For all the metrics mentioned above, if there are missing residues or atoms, we will input
the complete information into our model.
Because the ground truth structure doesn't include the coordinates of these components,
evaluating this type of data can be very challenging.
Fortunately, we know exactly which residues or atoms are missing, so we do not need to use any approximated alignment
when calculating these metrics.
We have found that using approximated alignments in metric calculations can sometimes result in
inaccurate metric values and hinder head-to-head comparisons between different methods.

### Performance

<img src="assets/img/hln.png" width="500" lt="AIDO Antibody/Nanobody Benchmark Results" width="500" />
<img src="assets/img/ana.png" width="500" alt="AIDO Antibody/Nanobody-Antigen by DcokQ" width="500" />

## Shell Completion

The CLI supports shell completion for bash, zsh, and fish. To enable completion, follow these instructions:

### Bash

Add this to your `~/.bashrc` or `~/.bash_profile`:

```bash
eval "$(_GENBIO_AIDOSP_COMPLETE=bash_source genbio-aidosp)"
```

### Zsh

Add this to your `~/.zshrc`:

```bash
eval "$(_GENBIO_AIDOSP_COMPLETE=zsh_source genbio-aidosp)"
```

### Fish

Add this to `~/.config/fish/completions/genbio-aidosp.fish`:

```fish
_GENBIO_AIDOSP_COMPLETE=fish_source genbio-aidosp | source
```

After adding these lines, restart your shell or source your configuration file.

> **_NOTE:_** Using `eval` means that the command is invoked and evaluated every time a shell is started, which can delay shell responsiveness. To speed it up, we ship pre-generated completion files. For example in `fish` you would do:

```bash
genbio-aidosp completion --shell fish > ~/.config/fish/completions/genbio-aidosp.fish
```


## License

Unless otherwise stated, this project is licensed under the GenBio AI Community License Agreement. This project includes third-party components ([MMseqs](https://github.com/soedinglab/MMseqs2), [Protenix](https://github.com/bytedance/Protenix)). Use of this project does not override or waive the original license terms of these third-party components - you are still bound by their respective licenses and can download from their original sites.


## Citation

Please cite AIDO.StructurePrediction using the following BibTex code:

```bibtex
@inproceedings{aido_structurepediction,
 title = {AIDO StructurePrediction},
 url = {https://huggingface.co/genbio-ai/AIDO.StructurePrediction},
 author = {Kun Leo, Jiayou Zhang, Georgy Andreev, Hugo Ly, Le Song, Eric P. Xing},
 year = {2025},
}
```
