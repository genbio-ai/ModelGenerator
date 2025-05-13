# Data

Data modules specify data sources, as well as data loading and preprocessing for use with [Tasks](../tasks).
They provide a simple interface for swapping data sources and re-using datasets for new workflows without any code changes, enabling rapid and reproducible experimentation.
They are specified with the `--data` arguent in the CLI or in the `data` section of a configuration file.

Data modules can automatically load common data sources (json, tsv, txt, HuggingFace) and uncommon ones (h5ad, TileDB).
They transform, split, and sample these sources for training with `mgen fit`, evaluation with `mgen test/validate`, and inference with `mgen predict`.

This reference overviews the available no-code data modules.
If you would like to develop new datasets, see [Experiment Design](../../experiment_design).

```yaml
data:
  class_path: modelgenerator.data.DMSFitnessPrediction
  init_args:
    path: genbio-ai/ProteinGYM-DMS
    train_split_files:
    - indels/B1LPA6_ECOSM_Russ_2020_indels.tsv
    train_split_name: train
    random_seed: 42
    batch_size: 32
    cv_num_folds: 5
    cv_test_fold_id: 0
    cv_enable_val_fold: true
    cv_fold_id_col: fold_id
model:
  ...
trainer:
  ...
```

> Note: Data modules are designed for use with a specific task, indicated in the class name.


## DNA

::: modelgenerator.data.NTClassification

::: modelgenerator.data.GUEClassification

::: modelgenerator.data.ClinvarRetrieve

::: modelgenerator.data.PromoterExpressionRegression

::: modelgenerator.data.PromoterExpressionGeneration

::: modelgenerator.data.DependencyMappingDataModule

## RNA

::: modelgenerator.data.TranslationEfficiency

::: modelgenerator.data.ExpressionLevel

::: modelgenerator.data.TranscriptAbundance

::: modelgenerator.data.ProteinAbundance

::: modelgenerator.data.NcrnaFamilyClassification

::: modelgenerator.data.SpliceSitePrediction

::: modelgenerator.data.ModificationSitePrediction

::: modelgenerator.data.RNAMeanRibosomeLoadDataModule

## Protein

::: modelgenerator.data.ContactPredictionBinary

::: modelgenerator.data.SspQ3

::: modelgenerator.data.FoldPrediction

::: modelgenerator.data.LocalizationPrediction

::: modelgenerator.data.MetalIonBinding

::: modelgenerator.data.SolubilityPrediction

::: modelgenerator.data.AntibioticResistance

::: modelgenerator.data.CloningClf

::: modelgenerator.data.MaterialProduction

::: modelgenerator.data.TcrPmhcAffinity

::: modelgenerator.data.PeptideHlaMhcAffinity

::: modelgenerator.data.TemperatureStability

::: modelgenerator.data.FluorescencePrediction

::: modelgenerator.data.FitnessPrediction

::: modelgenerator.data.StabilityPrediction

::: modelgenerator.data.EnzymeCatalyticEfficiencyPrediction

::: modelgenerator.data.OptimalTemperaturePrediction

::: modelgenerator.data.OptimalPhPrediction

::: modelgenerator.data.DMSFitnessPrediction

## Structure

::: modelgenerator.data.ContactPredictionBinary

::: modelgenerator.data.SspQ3

::: modelgenerator.data.FoldPrediction

::: modelgenerator.data.FluorescencePrediction

::: modelgenerator.data.DMSFitnessPrediction

::: modelgenerator.data.StructureTokenDataModule

## Cell

::: modelgenerator.data.CellClassificationDataModule

::: modelgenerator.data.CellClassificationLargeDataModule

::: modelgenerator.data.ClockDataModule

::: modelgenerator.data.PertClassificationDataModule

## Tissue

::: modelgenerator.data.CellWithNeighborDataModule

## Multimodal

::: modelgenerator.data.IsoformExpression

## Base Classes

::: modelgenerator.data.DataInterface

::: modelgenerator.data.ColumnRetrievalDataModule

::: modelgenerator.data.SequencesDataModule

::: modelgenerator.data.SequenceClassificationDataModule

::: modelgenerator.data.SequenceRegressionDataModule

::: modelgenerator.data.TokenClassificationDataModule

::: modelgenerator.data.DiffusionDataModule

::: modelgenerator.data.ClassDiffusionDataModule

::: modelgenerator.data.ConditionalDiffusionDataModule

::: modelgenerator.data.MLMDataModule
