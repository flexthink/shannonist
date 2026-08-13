# Conditional pairwise Gaussian MI

This recipe trains conditioned `PairwiseBA` on a uniform mixture of two
correlated-Gaussian regimes. A noisy Gaussian context identifies the regime,
so the flow learns `q(x_i | x_j, c)` and `H(x_i | c)`. Validation reports a
separate lower-bound matrix for each regime.

```bash
python recipes/PairwiseCorrelatedGaussian/CondMI/train.py \
  recipes/PairwiseCorrelatedGaussian/CondMI/hparams/train_ba.yaml
```
