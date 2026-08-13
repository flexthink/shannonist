# Conditional MI for latent pairwise correlated Gaussians

This recipe trains `SampledPairwiseBA` to estimate pairwise mutual information
conditioned on the latent context `z` emitted by
`LatentPairwiseCorrelatedGaussian`. The proposal models `q(x_i | x_j, z)` and
the entropy flow models `H(x_i | z)`.

The reported target is the dataset's exact `conditional_mutual_information`
matrix, which differs from its configured marginal MI matrix whenever
`context_fraction` is nonzero. The default fraction is deliberately small
(`0.05`), keeping the conditional targets on approximately the same scale as
the configured matrix while preserving a measurable role for `z`.

```bash
python recipes/LatentPairwiseCorrelatedGaussian/CondMI/train.py \
  recipes/LatentPairwiseCorrelatedGaussian/CondMI/hparams/train_ba.yaml
```

## Two-regime attention challenge

`train_mixture_ba.yaml` selects between two different latent correlated-
Gaussian MI matrices. The conditioning input is a masked bag of correlated
Gaussian tokens derived from the sample latent `z`. Regime-dependent means
are well separated, and `AttentionPoolingConditioning` reduces each bag to the
vector supplied to both conditional flows. Training and validation print both
learned matrices independently.

```bash
python recipes/LatentPairwiseCorrelatedGaussian/CondMI/train_mixture_ba.py \
  recipes/LatentPairwiseCorrelatedGaussian/CondMI/hparams/train_mixture_ba.yaml
```
