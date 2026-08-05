# Pairwise correlated Gaussian FLO recipe

This recipe jointly trains `PairwiseFLO` on vector-valued Gaussian variables
with a user-supplied pairwise mutual-information matrix. It prints the mean
learned MI lower-bound matrix beside the ground truth after every training and
validation stage.

Run the default 3-by-3 experiment with:

```bash
python recipes/PairwiseCorrelatedGaussian/FLO/train.py \
  recipes/PairwiseCorrelatedGaussian/FLO/hparams/train.yaml
```

Override the matrix and related hyperparameters from the command line:

```bash
python recipes/PairwiseCorrelatedGaussian/FLO/train.py \
  recipes/PairwiseCorrelatedGaussian/FLO/hparams/train.yaml \
  --mutual_information='[[0, 0.3, 0.1], [0.3, 0, 0.2], [0.1, 0.2, 0]]' \
  --number_of_epochs=5 --lr=0.0005
```

The MI matrix must be symmetric with a zero diagonal. Its implied Gaussian
correlation matrix must also be positive semidefinite. When changing the matrix
size, set `count` to the same dimension.
