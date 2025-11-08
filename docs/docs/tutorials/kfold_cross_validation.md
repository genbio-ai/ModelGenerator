# K-fold cross validation

Datasets implementing the `DataInterface` with the `KFoldMixin` support semi-automatic k-fold crossvalidation for uncertainty estimation.

We use translation efficiency prediction as an example task to demonstrate how to do a k-fold cross validation in ModelGenerator. The logic is to split the dataset into k-fold, and call each fold as a test set iteratively.

#### Data configs
For cross validation task, we input only one dataset named `train` containing a colomn `fold_id` indicating the fold index for each sample. You need to set `cv_num_folds`, `cv_test_fold_id`, `cv_enable_val_fold`, `cv_fold_id_col` according to your experiment setting.
```yaml
data:
  class_path: modelgenerator.data.TranslationEfficiency
  init_args:
    path: genbio-ai/rna-downstream-tasks
    config_name: translation_efficiency_Muscle
    normalize: true
    train_split_name: train
    random_seed: 42
    batch_size: 8
    shuffle: true
    cv_num_folds: 10
    cv_test_fold_id: 0
    cv_enable_val_fold: true
    cv_fold_id_col: fold_id
```
See `experiments/AIDO.RNA/configs/translation_efficiency.yaml` for full hyperparameter settings.


#### Finetuning script
```shell
for FOLD in {0..9}
  do
    RUN_NAME=te_Muscle_aido_rna_1b600m_fold${FOLD}
    CKPT_SAVE_DIR=logs/rna_tasks/${RUN_NAME}
    CUDA_VISIBLE_DEVICES=0 mgen fit --config experiments/AIDO.RNA/configs/translation_efficiency.yaml \
      --data.config_name translation_efficiency_Muscle \
      --data.cv_test_fold_id $FOLD \
      --trainer.logger.name $RUN_NAME \
      --trainer.callbacks.dirpath $CKPT_SAVE_DIR
  done
```

#### Evaluation script
```shell
for FOLD in {0..9}
do
  CKPT_PATH=logs/rna_tasks/te_Muscle_aido_rna_1b600m_fold${FOLD}/best_val*
  echo ">>> Fold ${FOLD}"
  mgen test --config experiments/AIDO.RNA/configs/translation_efficiency.yaml \
    --data.config_name translation_efficiency_Muscle \
    --data.cv_test_fold_id $FOLD \
    --model.strict_loading True \
    --model.reset_optimizer_states True \
    --trainer.logger null \
    --ckpt_path $CKPT_PATH
done
```
