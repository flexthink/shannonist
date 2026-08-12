# Shannonist

Shannonist is a PyTorch library for estimating and analyzing
information-theoretic properties of machine-learning models. Its estimators
are designed to be composable: they expose model predictions and objectives,
but do not own the optimizer or training loop.

The first implemented estimator is joint FLO (contrastive
Fenchel-Legendre optimization) for mutual information.

## Installation

Shannonist requires Python 3.10 or newer. Install the project in editable mode
while developing:

```bash
python -m pip install -e .
```

The runtime dependencies are PyTorch, TensorDict, and HyperPyYAML.

## Quick start: train a FLO estimator

The following example trains FLO on paired vectors. The encoders map both
variables into a shared feature space, and `BilinearPotential` learns the FLO
potential from their concatenated representations.

```python
import torch

from shannonist.mi import JointFLO, MIBatch
from shannonist.models import BilinearPotential, MLP

input_dim = 20
feature_dim = 64

estimator = JointFLO(
    encoder_x=MLP(input_dim, output_dim=feature_dim),
    encoder_y=MLP(input_dim, output_dim=feature_dim),
    potential=BilinearPotential(
        input_dim=feature_dim,
        hidden_dim=(128,),
    ),
    tau=1.0,
    use_norm=True,
)
optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-4)

x = torch.randn(256, input_dim)
y = torch.randn(256, input_dim)
batch = MIBatch(x=x, y=y, batch_size=[x.shape[0]])

estimator.train()
predictions = estimator.compute_forward(batch)
objective = estimator.compute_objectives(predictions)

optimizer.zero_grad()
objective.loss.backward()
optimizer.step()
```

`compute_forward()` executes the critic and returns a `JointFLOOutput`
TensorClass containing the critic representations and potential values.
`compute_objectives()` consumes those predictions without running the model a
second time. The returned `ObjectiveOutput` contains:

- `loss`: the differentiable scalar to minimize;
- `metrics["loss_vec"]`: per-example FLO losses;
- `metrics["similarity"]`: the pairwise similarity matrix;
- `metrics["u"]`: the learned potential values.

## Estimate mutual information

After training, call `estimate()` with an `MIBatch`:

```python
estimator.eval()
with torch.no_grad():
    result = estimator.estimate(batch)

print(result.value.item())
print(result.details["similarity"].shape)
```

`result.value` is the estimated mutual information. FLO requires at least two
paired samples and currently operates on two-dimensional feature matrices with
shape `(batch, features)`.

## Synthetic correlated Gaussian data

`CorrelatedGausian` is a lazy synthetic dataset with a known mutual information
in nats. Individual samples and collated batches are TensorDicts:

```python
from torch.utils.data import DataLoader

from shannonist.mi import CorrelatedGausian, tensordict_collate

dataset = CorrelatedGausian(
    mutual_information=2.0,
    dim=20,
    num_samples=10_000,
)
loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    collate_fn=tensordict_collate,
)

batch = next(iter(loader))
print(batch["x"].shape)  # torch.Size([256, 20])
print(batch["y"].shape)  # torch.Size([256, 20])
print(dataset.rho)
```

It samples

```text
X ~ N(0, I_d)
Y = rho X + sqrt(1 - rho^2) epsilon
rho = sqrt(1 - exp(-2 I* / d))
```

where `epsilon` is an independent standard Gaussian.

For experiments involving more than two variables, provide a symmetric
pairwise-MI matrix to `PairwiseCorrelatedGaussian`:

```python
import torch

from shannonist.mi import PairwiseCorrelatedGaussian

pairwise_mi = torch.tensor(
    [
        [0.0, 0.2, 0.1],
        [0.2, 0.0, 0.15],
        [0.1, 0.15, 0.0],
    ]
)
dataset = PairwiseCorrelatedGaussian(
    mutual_information=pairwise_mi,
    dim=20,
    num_samples=10_000,
)

sample = dataset[0]
print(sample["x"].shape)  # torch.Size([3, 20])
```

The dataset converts each MI value to an isotropic Gaussian correlation and
generates all variables from independent latent Gaussians. The resulting
correlation matrix must be positive semidefinite; inconsistent MI matrices are
rejected with `ValueError`.

## Run the included recipe

The correlated-Gaussian recipe provides a complete SpeechBrain-style training
loop without depending on SpeechBrain. Run its FLO configuration with:

```bash
python recipes/CorrelatedGaussian/MI/train.py \
  recipes/CorrelatedGaussian/MI/hparams/train_flo.yaml
```

The same script runs the Barber-Agakov estimator with:

```bash
python recipes/CorrelatedGaussian/MI/train.py \
  recipes/CorrelatedGaussian/MI/hparams/train_ba.yaml
```

The recipe is configured with HyperPyYAML. Override configuration values from
the command line:

```bash
python recipes/CorrelatedGaussian/MI/train.py \
  recipes/CorrelatedGaussian/MI/hparams/train_flo.yaml \
  --device cuda:0 \
  --mutual_information=1.0 \
  --number_of_epochs=5
```

VS Code users can launch the same recipe with the included
`FLO: Correlated Gaussian` debug configuration.

The pairwise recipe trains `PairwiseFLO` from a user-supplied MI matrix and
prints its learned lower bounds beside the ground truth:

```bash
python recipes/PairwiseCorrelatedGaussian/MI/train.py \
  recipes/PairwiseCorrelatedGaussian/MI/hparams/train_flo.yaml
```

See the recipe's README for matrix override examples and validity constraints.

## Development and tests

Install the test dependencies and run the unit suite with:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Pytest only collects the library tests under `tests/`; runnable recipe scripts
are intentionally excluded.

## The estimator interface

Every estimator implements `estimate(batch)`. A `TrainableEstimator` also
separates prediction from objective construction:

```python
predictions = estimator.compute_forward(batch)
objective = estimator.compute_objectives(predictions)
```

This separation lets a larger recipe reuse predictions, inspect intermediate
representations, combine multiple objectives, or manage optimization itself.
Shannonist also includes a small `Brain` abstraction for conventional training
loops, but using it is optional.

## Sequence masks

`MIBatch` can carry optional `x_mask` and `y_mask` tensors for sequence-valued
features:

```python
batch = MIBatch(
    x=x,                # (batch, x_length, features)
    y=y,                # (batch, y_length, features)
    x_mask=x_mask,      # (batch, x_length, mask)
    y_mask=y_mask,      # (batch, y_length, mask)
    batch_size=[x.shape[0]],
)
```

Nonzero mask values identify valid positions. The schema supports these masks,
but `JointFLO` does not yet implement masked estimation and raises
`NotImplementedError` when either mask is supplied. `PairwiseFLO` and
`PairwiseBA` support an `x_mask` shaped `(*, count)` or `(*, count, 1)`: for
pair `(i, j)`, a sample is completely excluded unless both mask positions are
valid.

## Project status

Shannonist is in early development. APIs may evolve as additional mutual-
information estimators and other information-theoretic quantities are added.
