# Pairwise correlated Gaussian MI recipes

These recipes jointly train either `PairwiseFLO` or `PairwiseBA` on
vector-valued Gaussian variables with a user-supplied pairwise
mutual-information matrix. They print the mean learned MI lower-bound matrix
beside the ground truth after every training and validation stage.

Run the default 3-by-3 experiment with:

```bash
python recipes/PairwiseCorrelatedGaussian/MI/train.py \
  recipes/PairwiseCorrelatedGaussian/MI/hparams/train_flo.yaml
```

Run Barber-Agakov with:

```bash
python recipes/PairwiseCorrelatedGaussian/MI/train.py \
  recipes/PairwiseCorrelatedGaussian/MI/hparams/train_ba.yaml
```

Override the matrix and related hyperparameters from the command line:

```bash
python recipes/PairwiseCorrelatedGaussian/MI/train.py \
  recipes/PairwiseCorrelatedGaussian/MI/hparams/train_flo.yaml \
  --mutual_information='[[0, 0.3, 0.1], [0.3, 0, 0.2], [0.1, 0.2, 0]]' \
  --number_of_epochs=5 --lr=0.0005
```

The MI matrix must be symmetric with a zero diagonal. Its implied Gaussian
correlation matrix must also be positive semidefinite. When changing the matrix
size, set `count` to the same dimension.
