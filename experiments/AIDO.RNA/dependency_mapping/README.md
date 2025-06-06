# Dependency Mapping

Dependency mapping is an _in silico_ mutagenesis technique that identifies co-conserved elements in a sequence.
AIDO.ModelGenerator implements the procedure proposed by [Tomaz da Silva et al.](https://www.biorxiv.org/content/10.1101/2024.07.27.605418v1)
We use this to assess structural features learned during pretraining in the [AIDO.RNA](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1) paper with the [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) model.
This task uses the pre-trained models directly, and does not require finetuning.

To reproduce the dependency mapping results from the AIDO.RNA paper, run the following from the ModelGenerator root directory:
```
# Inference
mgen predict --config experiments/AIDO.RNA/dependency_mapping/config.yaml

# Plotting
python experiments/AIDO.RNA/dependency_mapping/plot_dependency_maps.py \
    -i depmap_predictions \
    -o depmap_plots \
    -v experiments/AIDO.RNA/dependency_mapping/DNA.txt \
    -t modelgenerator/huggingface_models/rnabert/vocab.txt 
```

To create new dependency maps,

1. Gather your sequences into a .tsv file with an `id` and `sequence` column.
2. Run `mgen predict --config config.yaml` where
```
model:
  class_path: Inference
  init_args: 
    backbone: <you-choose>
data:
  class_path: DependencyMappingDataModule
  init_args:
    path: <path/to/your/seq/dir>  # Note: this errors for ., use ../dependency_mapping if necessary
    test_split_files: 
      - <my_sequences.tsv>
    vocab_file: <vocab>.txt  # E.g. experiments/AIDO.RNA/dependency_mapping/DNA.txt
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: predictions
      filetype: pt
```

3. Run the plotting tool
```
python experiments/AIDO.RNA/dependency_mapping/plot_dependency_maps.py \
    -i <prediction_dir> \
    -o <output_dir> \
    -v <vocab.txt> \
    -t <tokenizer_vocab.txt>  
```

The output will be files of the name `<id>.png` in the output directory, with heatmaps of dependencies and logos with sequence information content.