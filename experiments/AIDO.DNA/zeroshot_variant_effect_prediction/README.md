# Zeroshot Variant Effect Prediction

Zeroshot variant effect prediction refers to the task of predicting the functional impact of genetic variants, especially single nucleotide polymorphisms (SNPs), without requiring additional task-specific fine-tuning of the model.
AIDO.ModelGenerator implements the procedure proposed by [Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1)
We use this to predict the effects of single nucleotide polymorphisms (SNPs) in the [AIDO.DNA-300M](https://huggingface.co/genbio-ai/AIDO.DNA-300M) model.
This task uses the pre-trained models directly, and does not require finetuning.


1. `ClinvarRetrieve` Module automatically download human reference genome, processed clinvar data and demo clinvar data to `GENBIO_DATA_DIR/genbio_finetune/dna_datasets` from hugginface [https://huggingface.co/datasets/genbio-ai/Clinvar](https://huggingface.co/datasets/genbio-ai/Clinvar).

Note: check if you have already set `GENBIO_DATA_DIR` as an environment variable on terminal
```
echo $GENBIO_DATA_DIR
```
If not, set your own data path
```
echo 'export GENBIO_DATA_DIR=/your/full/path/' >> ~/.bashrc
source ~/.bashrc
```
Otherwise, `ClinvarRetrieve` will automatically set the `GENBIO_DATA_DIR` environment variable to `<root_path>/ModelGenerator/genbio_data`.

2. You can also choose to download raw data and preprocess clinvar data by yourself:

```
# download hg38 reference genome
wget -P $GENBIO_DATA_DIR/genbio_finetune/dna_datasets https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip $GENBIO_DATA_DIR/genbio_finetune/dna_datasets/hg38.fa.gz

# download raw clinvar dataset
wget -P $GENBIO_DATA_DIR/genbio_finetune/dna_datasets https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/clinvar/clinvarMain.bb

# download package to convert .bb file to .bed file
wget http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigBedToBed
chmod +x bigBedToBed
./bigBedToBed $GENBIO_DATA_DIR/genbio_finetune/dna_datasets/clinvarMain.bb $GENBIO_DATA_DIR/genbio_finetune/dna_datasets/clinvarMain.bed

# Preprocess the raw ClinVar dataset to retain only the following fields: 'chrom', 'start', 'end', 'name', '_clinSignCode', 'ref', 'mutate', and 'effect'.
# The processed file is saved as Clinvar_Processed.tsv.
cd ModelGenerator
python experiments/AIDO.DNA/zeroshot_variant_effect_prediction/preprocess_clinvar.py $GENBIO_DATA_DIR/genbio_finetune/dna_datasets/clinvarMain.bed
```


2. Run `mgen test --config config.yaml`.
If you want to take the norm distance between reference and variant sequence embeddings as prediction, the config should be
```
model:
  class_path: modelgenerator.tasks.ZeroshotPredictionDistance
  init_args:
    backbone: <you-choose>
data:
  class_path: ClinvarRetrieve
  init_args:
    method: Distance
    window: <window size centered around the SNPs>
    test_split_files:
      - <my_sequences.tsv>
    reference_file: <human_reference_genome.ml.fa> # Example: hg38.ml.fa
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: predictions
      filetype: tsv
      write_cols: ['score','norm_type','labels','num_layer']
```
If you take the loglikelihood ratio between reference and variant sequence embeddings at the mutation position as prediction, then the config should be
```
model:
  class_path: modelgenerator.tasks.ZeroshotPredictionDiff
  init_args:
    backbone: <you-choose>
data:
  class_path: ClinvarRetrieve
  init_args:
    method: Diff
    window: <window size centered around the SNPs>
    test_split_files:
      - <my_sequences.tsv>
    reference_file: <human_reference_genome.ml.fa> # Example: hg38.ml.fa
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: predictions
      filetype: tsv
      write_cols: ['score','label']
```
The labels and scores are also saved in `test_predictions.tsv` under the dir specified by `--trainer.callbacks.output_dir`.

Here are two examples of how to load HF model for inference
For norm distance mode
```
mgen test --config experiments/AIDO.DNA/zeroshot_variant_effect_prediction/Clinvar_300M_zeroshot_Distance.yaml
```
For loglikelihood ratio mode
```
mgen test --config experiments/AIDO.DNA/zeroshot_variant_effect_prediction/Clinvar_300M_zeroshot_Diff.yaml
```