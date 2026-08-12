# Correlated Gaussian MI recipes

These recipes train either the joint FLO or joint Barber-Agakov
mutual-information estimator on synthetic correlated Gaussian pairs with a
known target mutual information. Both configurations share one training
script.

From an editable installation of Shannonist, run FLO with:

```bash
python recipes/CorrelatedGaussian/MI/train.py \
  recipes/CorrelatedGaussian/MI/hparams/train_flo.yaml
```

Run Barber-Agakov with:

```bash
python recipes/CorrelatedGaussian/MI/train.py \
  recipes/CorrelatedGaussian/MI/hparams/train_ba.yaml
```

Hyperparameters can be overridden from the command line:

```bash
python recipes/CorrelatedGaussian/MI/train.py \
  recipes/CorrelatedGaussian/MI/hparams/train_ba.yaml \
  --device cuda:0 --mutual_information=1.0 --number_of_epochs=5
```
