FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04 AS build
WORKDIR /workspace
# TODO: using conda just to get a Python binary is probably overkill
RUN apt update && apt install -y wget git
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh && \
    bash miniconda3.sh -b -u -p /opt/conda
RUN /opt/conda/bin/conda create -y -n finetune python=3.10
ENV PATH=/opt/conda/envs/finetune/bin:$PATH

# TODO: change to git clone when repos are public
COPY modelgenerator modelgenerator
COPY pyproject.toml .
COPY README.md .

RUN pip install --upgrade pip
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
RUN pip install flash_attn==2.7.4.post1

## RNA and Protein inverse folding requirements
RUN pip install torch_geometric==2.6.1
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN pip install biopython==1.84
RUN pip install MDAnalysis==2.8.0
RUN pip install biotite==1.0.1
RUN pip install OmegaConf

WORKDIR /workspace
RUN pip install -e .

FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04
WORKDIR /workspace
COPY --from=build /opt/conda/envs /opt/conda/envs
ENV PATH=/opt/conda/envs/finetune/bin:$PATH
COPY modelgenerator modelgenerator
ENV MGEN_DATA_DIR=/mgen_data
RUN mkdir ${MGEN_DATA_DIR}
