# Correlated Gaussian FLO recipe

This recipe trains the bilinear FLO mutual-information estimator on synthetic
correlated Gaussian pairs with a known target mutual information.

From an editable installation of Shannonist, run:

```bash
python recipes/CorrelatedGaussian/FLO/train.py \
  recipes/CorrelatedGaussian/FLO/hparams/train.yaml
```

Hyperparameters can be overridden from the command line:

```bash
python recipes/CorrelatedGaussian/FLO/train.py \
  recipes/CorrelatedGaussian/FLO/hparams/train.yaml \
  --device cuda:0 --mutual_information=1.0 --number_of_epochs=5
```
