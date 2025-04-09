import unittest
from modelgenerator.backbones import (
    aido_dna_debug,
    aido_protein_debug,
    dna_onehot,
    protein_onehot,
    enformer,
    borzoi,
)
from modelgenerator.data import *
from modelgenerator.tasks import *
from functools import partial
from lightning import Trainer


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.sequence_adapter_partial = partial(LinearCLSAdapter)
        self.token_adapter_partial = partial(LinearAdapter)
        self.conditional_generation_adapter_partial = partial(ConditionalLMAdapter)
        self.trainer = Trainer(fast_dev_run=True)

    def _test(self, task, data):
        trainer = Trainer(fast_dev_run=True)
        trainer.fit(task, data)
        trainer.test(task, data)

    def _test_inference(self, task, data):
        trainer = Trainer(fast_dev_run=True)
        trainer.test(task, data)

    def test_NTClassification(self):
        data = NTClassification()
        task = SequenceClassification(
            backbone=aido_dna_debug,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_GUEClassification(self):
        data = GUEClassification()
        task = SequenceClassification(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_MLMDataModule(self):
        data = MLMDataModule(
            path="InstaDeepAI/nucleotide_transformer_downstream_tasks",
            config_name="enhancers",
        )
        task = MLM(
            backbone=aido_dna_debug,
        )
        self._test(task, data)

    def test_TranslationEfficiency(self):
        data = TranslationEfficiency(normalize=False)
        task = SequenceRegression(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_ExpressionLevel(self):
        data = ExpressionLevel(normalize=False)
        task = SequenceRegression(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_TranscriptAbundance(self):
        data = TranscriptAbundance(normalize=False)
        task = SequenceRegression(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_ProteinAbundance(self):
        data = ProteinAbundance(normalize=False)
        task = SequenceRegression(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_NcrnaFamilyClassification(self):
        data = NcrnaFamilyClassification()
        task = SequenceClassification(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
            n_classes=88,
        )
        self._test(task, data)

    def test_SpliceSitePrediction(self):
        data = SpliceSitePrediction()
        task = SequenceClassification(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_ModificationSitePrediction(self):
        data = ModificationSitePrediction()
        task = SequenceClassification(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
            n_classes=12,
            multilabel=True,
        )
        self._test(task, data)

    def test_PromoterExpressionRegression(self):
        data = PromoterExpressionRegression(
            train_split_files=["test.tsv"],  # Make it go fast
            test_split_files=["test.tsv"],
            normalize=False,
        )
        task = SequenceRegression(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_PromoterExpressionGeneration(self):
        data = PromoterExpressionGeneration(
            train_split_files=["test.tsv"],  # Make it go fast
            test_split_files=["test.tsv"],
            normalize=False,
        )
        task = ConditionalDiffusion(
            backbone=aido_dna_debug,
            adapter=self.conditional_generation_adapter_partial,
            use_legacy_adapter=True,
        )
        self._test(task, data)

    def test_DMSFitnessPrediction(self):
        data = DMSFitnessPrediction(normalize=False)
        task = SequenceRegression(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_Embed(self):
        data = NTClassification()
        task = Embed(
            backbone=dna_onehot,
        )
        self.trainer.predict(task, data)

    def test_Inference(self):
        data = NTClassification()
        task = Inference(
            backbone=dna_onehot,
        )
        self.trainer.predict(task, data)

    def test_ContactPredictionBinary(self):
        data = ContactPredictionBinary()
        task = PairwiseTokenClassification(
            backbone=protein_onehot,
            adapter=self.token_adapter_partial,
        )
        self._test(task, data)

    def test_SspQ3(self):
        data = SspQ3()
        task = TokenClassification(
            backbone=protein_onehot, adapter=self.token_adapter_partial, n_classes=3
        )
        self._test(task, data)

    def test_FoldPrediction(self):
        data = FoldPrediction()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
            n_classes=1195,
        )
        self._test(task, data)

    def test_LocalizationPrediction(self):
        data = LocalizationPrediction()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
            n_classes=10,
        )
        self._test(task, data)

    def test_MetalIonBinding(self):
        data = MetalIonBinding()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_SolubilityPrediction(self):
        data = SolubilityPrediction()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_AntibioticResistance(self):
        data = AntibioticResistance()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
            n_classes=19,
        )
        self._test(task, data)

    def test_CloningClf(self):
        data = CloningClf()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_MaterialProduction(self):
        data = MaterialProduction()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_TcrPmhcAffinity(self):
        data = TcrPmhcAffinity()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_PeptideHlaMhcAffinity(self):
        data = PeptideHlaMhcAffinity()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_TemperatureStability(self):
        data = TemperatureStability()
        task = SequenceClassification(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_DiffusionDataModule(self):
        data = DiffusionDataModule(
            path="InstaDeepAI/nucleotide_transformer_downstream_tasks",
            config_name="enhancers",
        )
        task = Diffusion(
            backbone=dna_onehot,
            adapter=self.token_adapter_partial,
            use_legacy_adapter=False,
        )
        self._test(task, data)

    def test_ClassDiffusionDataModule(self):
        data = ClassDiffusionDataModule(
            path="InstaDeepAI/nucleotide_transformer_downstream_tasks",
            config_name="enhancers",
            class_filter=1,
        )
        task = Diffusion(
            backbone=dna_onehot,
            adapter=self.token_adapter_partial,
            use_legacy_adapter=False,
        )
        self._test(task, data)

    def test_FluorescencePrediction(self):
        data = FluorescencePrediction(normalize=False)
        task = SequenceRegression(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_FitnessPrediction(self):
        data = FitnessPrediction(normalize=False)
        task = SequenceRegression(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_StabilityPrediction(self):
        data = StabilityPrediction(normalize=False)
        task = SequenceRegression(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_EnzymeCatalyticEfficiencyPrediction(self):
        data = EnzymeCatalyticEfficiencyPrediction(normalize=False)
        task = SequenceRegression(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_OptimalTemperaturePrediction(self):
        data = OptimalTemperaturePrediction(normalize=False)
        task = SequenceRegression(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_OptimalPhPrediction(self):
        data = OptimalPhPrediction(normalize=False)
        task = SequenceRegression(
            backbone=protein_onehot,
            adapter=self.sequence_adapter_partial,
        )
        self._test(task, data)

    def test_IsoformExpression(self):
        unimodal_data = IsoformExpression(train_split_files = ["train_1.tsv"], x_col = "dna_seq", normalize=False)
        unimodal_task = SequenceRegression(
            backbone=dna_onehot,
            adapter=self.sequence_adapter_partial,
            num_outputs=30
        )
        self._test(unimodal_task, unimodal_data)
        multimodal_data = IsoformExpression(train_split_files = ["train_1.tsv"], normalize=False)
        bimodal_task = MMSequenceRegression(
            backbone=dna_onehot,
            backbone1=dna_onehot,
            num_outputs=30
        )
        self._test(bimodal_task, multimodal_data)

        trimodal_task = MMSequenceRegression(
            backbone=dna_onehot,
            backbone1=dna_onehot,
            backbone2=protein_onehot,
            backbone_order=["dna_seq", "rna_seq", "protein_seq"],
            num_outputs=30
        )
        self._test(trimodal_task, multimodal_data)

    def test_clinvar(self):
        data = ClinvarRetrieve(
            test_split_files = ["ClinVar_demo.tsv"],
            batch_size = 2,
        )
        task = ZeroshotPredictionDistance(
            backbone=aido_dna_debug,
        )
        self._test_inference(task, data)

    def test_clinvar_diff(self):
        data = ClinvarRetrieve(
            test_split_files = ["ClinVar_demo.tsv"],
            batch_size = 2,
            method='Diff',
        )
        task = ZeroshotPredictionDiff(
            backbone=aido_dna_debug,
        )
        self._test_inference(task, data)

    def test_clinvar_enformer(self):
        data = ClinvarRetrieve(
            test_split_files = ["ClinVar_demo.tsv"],
            window=96000,
            batch_size = 2,
        )
        task = ZeroshotPredictionDistance(
            backbone=enformer,
        )
        self._test_inference(task, data)

    def test_enformer_embed(self):
        data = ClinvarRetrieve(
            test_split_files = ["ClinVar_demo.tsv"],
            window=96000,
            batch_size = 2,
            method='Diff',
        )
        task = Embed(
            backbone=enformer,
        )
        self.trainer.predict(task, data)

    def test_clinvar_borzoi(self):
        data = ClinvarRetrieve(
            test_split_files = ["ClinVar_demo.tsv"],
            window=262144,
            batch_size = 2,
        )
        task = ZeroshotPredictionDistance(
            backbone=borzoi,
        )
        self._test_inference(task, data)

    def test_borzoi_embed(self):
        data = ClinvarRetrieve(
            test_split_files = ["ClinVar_demo.tsv"],
            window=262144,
            batch_size = 2,
            method='Diff',
        )
        task = Embed(
            backbone=borzoi,
        )
        self.trainer.predict(task, data)

if __name__ == "__main__":
    unittest.main()
