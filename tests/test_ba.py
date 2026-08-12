import math

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from shannonist.framework import ObjectiveOutput
from shannonist.mi import (
    EntropyEstimator,
    GaussianEntropyEstimator,
    GaussianProposal,
    FlowEntropyEstimator,
    FlowProposal,
    MIBatch,
    JointBA,
    JointBAOutput,
    PairwiseBA,
    PairwiseBAOutput,
    PairwiseMIBatch,
    SampledPairwiseBA,
    SampledPairwiseBAOutput,
    Proposal,
    StandardNormalEntropyEstimator,
    make_entropy_estimator,
    make_proposal,
    joint_ba_loss,
    pairwise_ba_loss,
    sampled_pairwise_ba_loss,
)
from shannonist.models import (
    FlowDensityEstimator,
    InvertibleLinear,
    InvertibleMLP,
    MultiMLP,
)


class ConstantDensityProposal(Proposal):
    """Proposal returning a fixed density at every value."""

    def __init__(self, dim: int, density: float) -> None:
        super().__init__(dim)
        self.density = density

    def forward(self, condition: torch.Tensor) -> TensorDict:
        density = condition.new_full(condition.shape[:-1], self.density)
        return TensorDict({"density": density}, batch_size=condition.shape[:-1])

    def log_prob(
        self,
        x: torch.Tensor,
        params: TensorDict | None = None,
    ) -> torch.Tensor:
        if params is None:
            params = self(x)
        return params["density"].log()


class RecordingDensityProposal(ConstantDensityProposal):
    """Constant proposal recording the pair-block shapes it receives."""

    def __init__(self, dim: int, density: float) -> None:
        super().__init__(dim, density)
        self.condition_shapes: list[torch.Size] = []

    def forward(self, condition: torch.Tensor) -> TensorDict:
        self.condition_shapes.append(condition.shape)
        return super().forward(condition)


class ConstantEntropyEstimator(EntropyEstimator):
    """Entropy estimator returning one learned constant."""

    def __init__(self, dim: int, entropy: float) -> None:
        super().__init__(dim)
        self.entropy = nn.Parameter(torch.tensor(entropy))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_ones(x.shape[:-1]) * self.entropy


class TrainableConstantEntropyEstimator(ConstantEntropyEstimator):
    """Constant entropy estimator with a fitting objective."""

    def compute_objectives(
        self,
        predictions: torch.Tensor,
    ) -> ObjectiveOutput:
        mean = predictions.mean()
        return ObjectiveOutput(loss=mean, estimate=mean, batch_size=[])


def test_gaussian_proposal_shapes_and_initialization() -> None:
    proposal = GaussianProposal(dim=3)
    x = torch.randn(128, 3)

    params = proposal(x)
    mu = params["mu"]
    logvar = params["logvar"]

    assert mu.shape == logvar.shape == (128, 3)
    assert params.batch_size == torch.Size([128])
    assert torch.allclose(proposal.mu.bias, torch.zeros_like(proposal.mu.bias))
    assert torch.allclose(mu.mean(dim=0), torch.zeros(3), atol=0.2)
    assert torch.equal(logvar, torch.zeros_like(logvar))
    assert torch.equal(torch.exp(0.5 * logvar), torch.ones_like(logvar))


def test_gaussian_proposal_parameters_are_differentiable() -> None:
    proposal = GaussianProposal(dim=3)
    x = torch.randn(5, 3)

    params = proposal(x)
    (params["mu"].sum() + params["logvar"].sum()).backward()

    assert proposal.mu.weight.grad is not None
    assert proposal.logvar.weight.grad is not None


def test_gaussian_proposal_probability_density() -> None:
    proposal = GaussianProposal(dim=2)
    params = proposal(torch.zeros(3, 2))

    density = proposal.prob(torch.zeros(3, 2), params)

    expected = torch.full((3,), 1 / (2 * torch.pi))
    assert torch.allclose(density, expected)


def test_gaussian_proposal_log_probability_stays_in_log_space() -> None:
    proposal = GaussianProposal(dim=256)
    x = torch.zeros(2, 256)
    params = proposal(x)

    log_prob = proposal.log_prob(x, params)

    expected = torch.full((2,), -128 * torch.log(torch.tensor(2 * torch.pi)))
    assert torch.allclose(log_prob, expected)
    assert torch.all(torch.isfinite(log_prob))


def test_gaussian_proposal_probability_accepts_arbitrary_values() -> None:
    proposal = GaussianProposal(dim=1)
    params = TensorDict(
        {"mu": torch.zeros(2, 1), "logvar": torch.zeros(2, 1)},
        batch_size=[2],
    )
    x = torch.tensor([[0.0], [1.0]])

    density = proposal.prob(x, params)

    expected = torch.exp(-(x.squeeze(-1) ** 2) / 2) / torch.sqrt(
        torch.tensor(2 * torch.pi)
    )
    assert torch.allclose(density, expected)


def test_gaussian_proposal_probability_computes_optional_params() -> None:
    proposal = GaussianProposal(dim=2)
    x = torch.zeros(3, 2)

    density = proposal.prob(x)

    expected = torch.full((3,), 1 / (2 * torch.pi))
    assert torch.allclose(density, expected)


def test_gaussian_proposal_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="dim"):
        GaussianProposal(dim=0)

    proposal = GaussianProposal(dim=3)
    params = proposal(torch.randn(4, 3))
    with pytest.raises(ValueError, match="trailing dimension"):
        proposal.prob(torch.randn(4, 2), params)


def test_gaussian_proposal_clamps_logvar() -> None:
    proposal = GaussianProposal(dim=2, min_logvar=-5.0, max_logvar=3.0)
    x = torch.zeros(4, 2)

    with torch.no_grad():
        proposal.logvar.bias.fill_(100.0)
    assert torch.equal(proposal(x)["logvar"], torch.full((4, 2), 3.0))

    with torch.no_grad():
        proposal.logvar.bias.fill_(-100.0)
    assert torch.equal(proposal(x)["logvar"], torch.full((4, 2), -5.0))


def test_gaussian_proposal_validates_logvar_bounds() -> None:
    with pytest.raises(ValueError, match="min_logvar"):
        GaussianProposal(dim=2, min_logvar=1.0, max_logvar=1.0)


def test_proposal_is_abstract() -> None:
    class IncompleteProposal(Proposal):
        def forward(self, condition: torch.Tensor) -> TensorDict:
            raise NotImplementedError

    with pytest.raises(TypeError):
        IncompleteProposal(dim=2)


def test_standard_normal_entropy_estimator() -> None:
    estimator = StandardNormalEntropyEstimator(dim=3)
    x = torch.randn(5, 3)

    entropy = estimator(x)

    expected = 3 / 2 * (1 + torch.log(torch.tensor(2 * torch.pi)))
    assert entropy.shape == (5,)
    assert torch.allclose(entropy, expected.expand(5))


def test_gaussian_entropy_estimator_fits_diagonal_gaussian() -> None:
    estimator = GaussianEntropyEstimator(dim=2)
    x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])

    entropy = estimator(x)

    expected = torch.log(torch.tensor(2 * torch.pi * math.e))
    assert estimator.count.item() == 2
    assert torch.equal(estimator.mean, torch.tensor([0.0, 1.0]))
    assert torch.equal(estimator.variance, torch.ones(2))
    assert torch.allclose(entropy, expected.expand(2))


def test_gaussian_entropy_estimator_accumulates_batches() -> None:
    x = torch.tensor(
        [[-2.0, 1.0], [0.0, 3.0], [4.0, -1.0], [2.0, 5.0]]
    )
    accumulated = GaussianEntropyEstimator(dim=2)
    accumulated.update(x[:2])
    accumulated.update(x[2:])
    reference = GaussianEntropyEstimator(dim=2)
    reference.update(x)

    assert accumulated.count.item() == 4
    assert torch.allclose(accumulated.mean, reference.mean)
    assert torch.allclose(accumulated.m2, reference.m2)
    assert torch.allclose(accumulated.variance, reference.variance)


def test_gaussian_entropy_estimator_does_not_update_during_eval() -> None:
    estimator = GaussianEntropyEstimator(dim=2)
    estimator(torch.randn(8, 2))
    count = estimator.count.clone()
    mean = estimator.mean.clone()
    variance = estimator.variance.clone()

    estimator.eval()
    entropy = estimator(torch.full((4, 2), 100.0))

    assert entropy.shape == (4,)
    assert torch.equal(estimator.count, count)
    assert torch.equal(estimator.mean, mean)
    assert torch.equal(estimator.variance, variance)


def test_gaussian_entropy_estimator_has_no_parameters() -> None:
    estimator = GaussianEntropyEstimator(dim=2)

    assert list(estimator.parameters()) == []


def test_gaussian_entropy_estimator_validates_state_and_input() -> None:
    with pytest.raises(ValueError, match="min_variance"):
        GaussianEntropyEstimator(dim=2, min_variance=0.0)

    estimator = GaussianEntropyEstimator(dim=2)
    with pytest.raises(ValueError, match="trailing dimension"):
        estimator(torch.randn(4, 3))

    estimator.eval()
    with pytest.raises(RuntimeError, match="no samples"):
        estimator(torch.randn(4, 2))


def test_gaussian_entropy_estimator_reset() -> None:
    estimator = GaussianEntropyEstimator(dim=2)
    estimator(torch.randn(4, 2))

    estimator.reset()

    assert estimator.count.item() == 0
    assert torch.equal(estimator.mean, torch.zeros(2))
    assert torch.equal(estimator.m2, torch.zeros(2))


def test_entropy_estimator_is_abstract() -> None:
    class IncompleteEntropyEstimator(EntropyEstimator):
        pass

    with pytest.raises(TypeError):
        IncompleteEntropyEstimator(dim=2)


def test_entropy_estimators_have_optional_objectives() -> None:
    estimator = StandardNormalEntropyEstimator(dim=2)
    predictions = estimator(torch.randn(4, 2))

    assert estimator.compute_objectives(predictions) is None


def test_flow_entropy_estimator_converts_log_density_to_entropy() -> None:
    density_estimator = FlowDensityEstimator(InvertibleLinear(dim=2))
    estimator = FlowEntropyEstimator(density_estimator)
    x = torch.zeros(3, 2)

    entropy = estimator(x)

    expected = torch.full((3,), torch.log(torch.tensor(2 * torch.pi)))
    assert estimator.dim == 2
    assert torch.allclose(entropy, expected, atol=1e-6)


def test_flow_entropy_estimator_objective_fits_only_density() -> None:
    density_estimator = FlowDensityEstimator(InvertibleLinear(dim=2))
    estimator = FlowEntropyEstimator(density_estimator)
    x = torch.randn(8, 2, requires_grad=True)

    predictions = estimator(x)
    objective = estimator.compute_objectives(predictions)
    objective.loss.backward()

    assert torch.allclose(objective.loss, predictions.mean())
    assert x.grad is None
    assert density_estimator.transform.log_diag.grad is not None


def test_flow_entropy_estimator_validates_density_estimator() -> None:
    with pytest.raises(TypeError, match="FlowDensityEstimator"):
        FlowEntropyEstimator(nn.Identity())


def test_flow_proposal_scores_pairwise_residuals() -> None:
    density_estimator = FlowDensityEstimator(InvertibleLinear(dim=2))
    proposal = FlowProposal(density_estimator, location_model=nn.Identity())
    y = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    x = torch.tensor([[2.0, 0.0], [3.0, 4.0]])

    params = proposal(y)
    log_prob = proposal.log_prob(x, params)

    assert params.batch_size == torch.Size([2])
    assert torch.equal(params["location"], y)
    assert torch.allclose(log_prob, density_estimator.log_prob(x - y))


def test_flow_proposal_probability_matches_log_probability() -> None:
    proposal = FlowProposal(
        FlowDensityEstimator(InvertibleLinear(dim=2))
    )
    x = torch.randn(4, 2)
    y = torch.randn(4, 2)
    params = proposal(y)

    assert torch.allclose(
        proposal.prob(x, params),
        proposal.log_prob(x, params).exp(),
    )


def test_flow_proposal_is_trainable_through_ba_objective() -> None:
    proposal = FlowProposal(
        FlowDensityEstimator(InvertibleLinear(dim=2))
    )
    estimator = JointBA(
        dim=2,
        enc_dim=2,
        encoder=nn.Identity(),
        conditional_proposal=proposal,
    )
    batch = MIBatch(
        x=torch.randn(8, 2),
        y=torch.randn(8, 2),
        batch_size=[8],
    )

    predictions = estimator.compute_forward(batch)
    estimator.compute_objectives(predictions).loss.backward()

    assert proposal.density_estimator.transform.log_diag.grad is not None
    assert proposal.location_model.weight.grad is not None


def test_flow_proposal_learns_location_from_y() -> None:
    proposal = FlowProposal(
        FlowDensityEstimator(InvertibleLinear(dim=2))
    )
    y = torch.randn(4, 2)

    params = proposal(y)

    assert isinstance(proposal.location_model, nn.Linear)
    assert params["location"].shape == y.shape
    assert torch.equal(
        proposal.location_model.bias,
        torch.zeros_like(proposal.location_model.bias),
    )


def test_flow_proposal_validates_density_estimator_and_shapes() -> None:
    with pytest.raises(TypeError, match="FlowDensityEstimator"):
        FlowProposal(nn.Identity())
    with pytest.raises(TypeError, match="location_model"):
        FlowProposal(
            FlowDensityEstimator(InvertibleLinear(dim=2)),
            location_model=object(),
        )

    proposal = FlowProposal(
        FlowDensityEstimator(InvertibleLinear(dim=2))
    )
    with pytest.raises(ValueError, match="y.*trailing dimension"):
        proposal(torch.randn(3, 4))
    with pytest.raises(ValueError, match="x.*trailing dimension"):
        proposal.log_prob(torch.randn(3, 4), proposal(torch.randn(3, 2)))


def test_entropy_estimator_factory() -> None:
    standard = make_entropy_estimator("standard_normal", dim=3)
    gaussian = make_entropy_estimator(
        "gaussian",
        dim=3,
        opts={"min_variance": 1e-5},
    )
    flow = make_entropy_estimator("flow", dim=3)

    assert isinstance(standard, StandardNormalEntropyEstimator)
    assert isinstance(gaussian, GaussianEntropyEstimator)
    assert gaussian.min_variance == 1e-5
    assert isinstance(flow, FlowEntropyEstimator)
    assert isinstance(flow.density_estimator.transform, InvertibleMLP)
    assert flow.density_estimator.transform.hidden_dims == (3,)


def test_proposal_factory() -> None:
    gaussian = make_proposal(
        "gaussian",
        dim=3,
        opts={"min_logvar": -4.0, "max_logvar": 4.0},
    )
    flow = make_proposal("flow", dim=3)

    assert isinstance(gaussian, GaussianProposal)
    assert gaussian.min_logvar == -4.0
    assert gaussian.max_logvar == 4.0
    assert isinstance(flow, FlowProposal)
    assert isinstance(flow.location_model, nn.Linear)
    assert isinstance(flow.density_estimator.transform, InvertibleMLP)
    assert flow.density_estimator.transform.hidden_dims == (3,)


def test_flow_proposal_factory_accepts_location_model() -> None:
    location_model = nn.Identity()

    proposal = make_proposal(
        "flow",
        dim=3,
        opts={"location_model": location_model},
    )

    assert proposal.location_model is location_model


def test_factories_validate_names_and_options() -> None:
    with pytest.raises(ValueError, match="unknown entropy"):
        make_entropy_estimator("mystery", dim=2)
    with pytest.raises(ValueError, match="unknown proposal"):
        make_proposal("mystery", dim=2)
    with pytest.raises(ValueError, match="unknown flow options"):
        make_proposal("flow", dim=2, opts={"mystery": True})


def test_joint_ba_accepts_proposals_and_encoder() -> None:
    conditional = GaussianProposal(dim=3)
    entropy_estimator = ConstantEntropyEstimator(dim=3, entropy=1.0)
    encoder = nn.Linear(2, 3)
    estimator = JointBA(
        dim=2,
        enc_dim=3,
        conditional_proposal=conditional,
        entropy_estimator=entropy_estimator,
        encoder=encoder,
    )
    batch = MIBatch(
        x=torch.randn(4, 2),
        y=torch.randn(4, 2),
        batch_size=[4],
    )

    assert estimator.dim == 2
    assert estimator.enc_dim == 3
    assert estimator.encoder is encoder
    assert estimator.conditional_proposal is conditional
    assert estimator.entropy_estimator is entropy_estimator
    estimate = estimator.estimate(batch)
    assert estimate.value.ndim == 0
    assert torch.isfinite(estimate.value)
    assert estimate.entropies.shape == (4,)


def test_joint_ba_bound_combines_entropy_and_conditional_density() -> None:
    estimator = JointBA(
        dim=1,
        enc_dim=1,
        encoder=nn.Identity(),
        conditional_proposal=ConstantDensityProposal(1, density=0.5),
        entropy_estimator=ConstantEntropyEstimator(
            1,
            entropy=torch.log(torch.tensor(4.0)).item(),
        ),
    )
    batch = MIBatch(
        x=torch.randn(4, 1),
        y=torch.randn(4, 1),
        batch_size=[4],
    )

    predictions = estimator.compute_forward(batch)
    objective = estimator.compute_objectives(predictions)
    estimate = estimator.estimate(batch)

    expected = torch.log(torch.tensor(2.0))
    assert torch.allclose(objective.loss, -expected)
    assert torch.allclose(estimate.value, expected)
    assert torch.allclose(
        objective.metrics["entropy_vec"],
        torch.full((4,), -torch.log(torch.tensor(0.25))),
    )
    assert torch.allclose(
        objective.metrics["conditional_log_prob"],
        torch.full((4,), torch.log(torch.tensor(0.5))),
    )


def test_joint_ba_adds_optional_entropy_objective() -> None:
    entropy = torch.log(torch.tensor(4.0))
    entropy_estimator = TrainableConstantEntropyEstimator(
        dim=4,
        entropy=entropy.item(),
    )
    estimator = JointBA(
        dim=4,
        enc_dim=4,
        encoder=nn.Identity(),
        conditional_proposal=ConstantDensityProposal(4, density=0.5),
        entropy_estimator=entropy_estimator,
    )
    batch = MIBatch(
        x=torch.randn(4, 4),
        y=torch.randn(4, 4),
        batch_size=[4],
    )

    predictions = estimator.compute_forward(batch)
    objective = estimator.compute_objectives(predictions)
    estimate = estimator.estimate(batch)

    expected_estimate = torch.log(torch.tensor(2.0))
    assert torch.allclose(estimate.value, expected_estimate)
    assert torch.allclose(objective.metrics["ba_loss"], -expected_estimate)
    assert torch.allclose(objective.metrics["entropy_loss"], entropy)
    assert torch.allclose(
        objective.loss,
        (entropy - expected_estimate) / estimator.enc_dim,
    )
    assert torch.allclose(objective.estimate, expected_estimate)


def test_joint_ba_loss_uses_negative_mean_bound() -> None:
    predictions = JointBAOutput(
        hx=torch.zeros(2, 1),
        hy=torch.zeros(2, 1),
        conditional_log_prob=torch.log(torch.tensor([0.5, 0.25])),
        entropy=torch.tensor([1.0, 2.0]),
        conditional_params=TensorDict({}, batch_size=[2]),
        batch_size=[2],
    )

    loss, details = joint_ba_loss(predictions)

    expected_vec = -(
        predictions.entropy + predictions.conditional_log_prob
    )
    assert torch.allclose(details["loss_vec"], expected_vec)
    assert torch.allclose(loss, expected_vec.mean())


def test_joint_ba_objective_backpropagates() -> None:
    entropy_estimator = ConstantEntropyEstimator(dim=2, entropy=1.0)
    estimator = JointBA(
        dim=2,
        enc_dim=2,
        entropy_estimator=entropy_estimator,
    )
    batch = MIBatch(
        x=torch.randn(8, 2),
        y=torch.randn(8, 2),
        batch_size=[8],
    )

    predictions = estimator.compute_forward(batch)
    objective = estimator.compute_objectives(predictions)
    objective.loss.backward()

    assert estimator.encoder.weight.grad is not None
    assert estimator.conditional_proposal.mu.weight.grad is not None
    assert entropy_estimator.entropy.grad is None


def test_joint_ba_rejects_masks() -> None:
    estimator = JointBA(dim=2, enc_dim=2)
    batch = MIBatch(
        x=torch.randn(4, 2),
        y=torch.randn(4, 2),
        x_mask=torch.ones(4, 1, dtype=torch.bool),
        batch_size=[4],
    )

    with pytest.raises(NotImplementedError, match="masks"):
        estimator.compute_forward(batch)


def test_joint_ba_builds_default_modules() -> None:
    estimator = JointBA(dim=2, enc_dim=3)

    assert isinstance(estimator.encoder, nn.Linear)
    assert estimator.encoder.in_features == 2
    assert estimator.encoder.out_features == 3
    assert isinstance(estimator.conditional_proposal, GaussianProposal)
    assert estimator.conditional_proposal.dim == 3
    assert isinstance(estimator.entropy_estimator, GaussianEntropyEstimator)
    assert estimator.entropy_estimator.dim == 3


def test_joint_ba_builds_named_flow_components() -> None:
    estimator = JointBA(
        dim=3,
        enc_dim=3,
        encoder=nn.Identity(),
        conditional_proposal="flow",
        entropy_estimator="flow",
    )

    assert isinstance(estimator.conditional_proposal, FlowProposal)
    assert isinstance(estimator.entropy_estimator, FlowEntropyEstimator)
    assert isinstance(
        estimator.conditional_proposal.density_estimator.transform,
        InvertibleMLP,
    )
    assert isinstance(
        estimator.entropy_estimator.density_estimator.transform,
        InvertibleMLP,
    )


def test_joint_ba_passes_factory_options() -> None:
    estimator = JointBA(
        dim=2,
        enc_dim=2,
        conditional_proposal="gaussian",
        entropy_estimator="gaussian",
        proposal_opts={"min_logvar": -3.0, "max_logvar": 2.0},
        estimator_opts={"min_variance": 1e-4},
    )

    assert estimator.conditional_proposal.min_logvar == -3.0
    assert estimator.conditional_proposal.max_logvar == 2.0
    assert estimator.entropy_estimator.min_variance == 1e-4


def test_joint_ba_rejects_options_for_instances() -> None:
    with pytest.raises(ValueError, match="proposal_opts"):
        JointBA(
            dim=2,
            enc_dim=2,
            conditional_proposal=GaussianProposal(2),
            proposal_opts={"min_logvar": -3.0},
        )
    with pytest.raises(ValueError, match="estimator_opts"):
        JointBA(
            dim=2,
            enc_dim=2,
            entropy_estimator=GaussianEntropyEstimator(2),
            estimator_opts={"min_variance": 1e-4},
        )


def test_joint_ba_rejects_plain_component_modules() -> None:
    with pytest.raises(TypeError, match="Proposal"):
        JointBA(dim=2, enc_dim=2, conditional_proposal=nn.Identity())
    with pytest.raises(TypeError, match="EntropyEstimator"):
        JointBA(dim=2, enc_dim=2, entropy_estimator=nn.Identity())


def test_joint_ba_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="dim"):
        JointBA(dim=0, enc_dim=2)
    with pytest.raises(ValueError, match="enc_dim"):
        JointBA(dim=2, enc_dim=0)
    with pytest.raises(ValueError, match="conditional_proposal.dim"):
        JointBA(
            dim=2,
            enc_dim=3,
            conditional_proposal=GaussianProposal(dim=2),
        )
    with pytest.raises(ValueError, match="entropy_estimator.dim"):
        JointBA(
            dim=2,
            enc_dim=3,
            entropy_estimator=StandardNormalEntropyEstimator(dim=2),
        )


def make_pairwise_ba(count: int = 3) -> PairwiseBA:
    """Construct a deterministic pairwise BA estimator for tests."""
    return PairwiseBA(
        dim=2,
        enc_dim=2,
        count=count,
        encoder=nn.Identity(),
        conditional_proposal=ConstantDensityProposal(2, density=0.5),
        entropy_estimator=ConstantEntropyEstimator(
            2,
            entropy=torch.log(torch.tensor(4.0)).item(),
        ),
    )


def test_pairwise_ba_returns_symmetric_matrix_and_zero_diagonal() -> None:
    estimator = make_pairwise_ba()
    batch = PairwiseMIBatch(
        x=torch.randn(8, 3, 2),
        batch_size=[8],
    )

    predictions = estimator.compute_forward(batch)
    objective = estimator.compute_objectives(predictions)
    estimate = estimator.estimate(batch)

    expected = torch.full((3, 3), torch.log(torch.tensor(2.0)))
    expected.fill_diagonal_(0)
    assert isinstance(predictions, PairwiseBAOutput)
    assert predictions.conditional_log_prob.shape == (8, 3, 3)
    assert predictions.entropy.shape == (8, 3)
    assert predictions.mask.shape == (8, 3)
    assert torch.allclose(objective.estimate, expected)
    assert torch.allclose(estimate.value, expected)
    assert estimate.entropies.shape == (8, 3)
    raw_loss = -expected.triu(1).sum() / 3
    assert torch.allclose(objective.loss, raw_loss / estimator.enc_dim)


def test_pairwise_ba_loss_averages_both_directions() -> None:
    conditional_log_prob = torch.tensor(
        [
            [
                [0.0, -1.0, -2.0],
                [-3.0, 0.0, -4.0],
                [-5.0, -6.0, 0.0],
            ],
            [
                [0.0, -2.0, -3.0],
                [-4.0, 0.0, -5.0],
                [-6.0, -7.0, 0.0],
            ],
        ]
    )
    entropy = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    predictions = PairwiseBAOutput(
        hx=torch.zeros(2, 3, 1),
        conditional_log_prob=conditional_log_prob,
        entropy=entropy,
        conditional_params=TensorDict({}, batch_size=[2, 3, 3]),
        mask=torch.ones(2, 3, dtype=torch.bool),
        batch_size=[2],
    )

    loss, details = pairwise_ba_loss(predictions)

    directed = entropy.unsqueeze(-1) + conditional_log_prob
    expected = ((directed + directed.transpose(-1, -2)) / 2).mean(dim=0)
    expected.fill_diagonal_(0)
    assert torch.allclose(details["estimate_matrix"], expected)
    assert torch.allclose(loss, -expected.triu(1).sum() / 3)


def test_pairwise_ba_mask_matches_jointly_valid_samples() -> None:
    estimator = make_pairwise_ba(count=2)
    x = torch.randn(5, 2, 2)
    mask = torch.tensor(
        [[1, 1], [1, 0], [1, 1], [0, 1], [1, 1]],
        dtype=torch.bool,
    )
    modified_x = x.clone()
    modified_x[~mask] = 1_000_000

    original = PairwiseMIBatch(x=x, x_mask=mask, batch_size=[5])
    modified = PairwiseMIBatch(
        x=modified_x,
        x_mask=mask.unsqueeze(-1),
        batch_size=[5],
    )
    original_objective = estimator.compute_objectives(
        estimator.compute_forward(original)
    )
    modified_objective = estimator.compute_objectives(
        estimator.compute_forward(modified)
    )

    assert torch.allclose(original_objective.loss, modified_objective.loss)
    assert original_objective.metrics["valid_counts"][0, 1] == 3


def test_pairwise_ba_default_components_and_gradients() -> None:
    estimator = PairwiseBA(dim=2, enc_dim=3, count=3)
    batch = PairwiseMIBatch(
        x=torch.randn(10, 3, 2),
        batch_size=[10],
    )

    objective = estimator.compute_objectives(estimator.compute_forward(batch))
    objective.loss.backward()

    assert isinstance(estimator.encoder, MultiMLP)
    assert isinstance(estimator.conditional_proposal, GaussianProposal)
    assert isinstance(estimator.entropy_estimator, GaussianEntropyEstimator)
    assert estimator.conditional_proposal.mu.weight.grad is not None
    assert next(estimator.encoder.parameters()).grad is not None


def test_pairwise_ba_validates_count_mask_and_pair_coverage() -> None:
    with pytest.raises(ValueError, match="at least two"):
        PairwiseBA(dim=2, enc_dim=2, count=1)

    estimator = make_pairwise_ba()
    invalid_mask = PairwiseMIBatch(
        x=torch.randn(4, 3, 2),
        x_mask=torch.ones(4, 2, dtype=torch.bool),
        batch_size=[4],
    )
    with pytest.raises(ValueError, match="x_mask"):
        estimator.compute_forward(invalid_mask)

    missing_pair = PairwiseMIBatch(
        x=torch.randn(4, 3, 2),
        x_mask=torch.tensor(
            [[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0]],
            dtype=torch.bool,
        ),
        batch_size=[4],
    )
    with pytest.raises(ValueError, match="no jointly valid samples"):
        estimator.compute_objectives(estimator.compute_forward(missing_pair))


def make_sampled_pairwise_ba(sample_size: int = 3) -> SampledPairwiseBA:
    """Construct a deterministic sampled pairwise BA estimator."""
    return SampledPairwiseBA(
        dim=2,
        enc_dim=2,
        sample_size=sample_size,
        encoder=nn.Identity(),
        conditional_proposal=ConstantDensityProposal(2, density=0.5),
        entropy_estimator=ConstantEntropyEstimator(
            2,
            entropy=torch.log(torch.tensor(4.0)).item(),
        ),
    )


def test_sampled_pairwise_ba_samples_only_valid_pairs() -> None:
    torch.manual_seed(7)
    estimator = make_sampled_pairwise_ba(sample_size=4)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    batch = PairwiseMIBatch(
        x=torch.randn(3, 6, 2),
        x_mask=mask.unsqueeze(-1),
        batch_size=[3],
    )

    predictions = estimator.compute_forward(batch)
    objective = estimator.compute_objectives(predictions)

    assert isinstance(predictions, SampledPairwiseBAOutput)
    assert predictions.batch_size == torch.Size([3, 3])
    assert predictions.hx.shape == (3, 3, 2, 2)
    assert predictions.conditional_log_prob.shape == (3, 3, 2)
    assert predictions.entropy.shape == (3, 3, 2)
    assert predictions.position_indices.shape == (3, 3, 2)
    selected_mask = mask[
        torch.arange(3)[:, None, None],
        predictions.position_indices,
    ]
    assert selected_mask.all()
    assert torch.all(
        predictions.position_indices[..., 0]
        != predictions.position_indices[..., 1]
    )
    expected = torch.log(torch.tensor(2.0))
    assert torch.allclose(objective.estimate, expected)
    assert torch.allclose(objective.metrics["estimate_vec"], expected)
    assert torch.allclose(objective.metrics["ba_loss"], -expected)
    assert torch.allclose(objective.loss, -expected / estimator.enc_dim)


def test_sampled_pairwise_ba_loss_averages_directions_and_samples() -> None:
    entropy = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    log_prob = torch.tensor(
        [
            [[-1.0, -3.0], [-2.0, -4.0]],
            [[-5.0, -7.0], [-6.0, -8.0]],
        ]
    )
    predictions = SampledPairwiseBAOutput(
        hx=torch.zeros(2, 2, 2, 1),
        conditional_log_prob=log_prob,
        entropy=entropy,
        conditional_params=TensorDict({}, batch_size=[2, 2, 2]),
        position_indices=torch.tensor(
            [[[0, 1], [1, 2]], [[0, 1], [1, 2]]]
        ),
        batch_size=[2, 2],
    )

    loss, details = sampled_pairwise_ba_loss(predictions)

    expected_vec = (entropy + log_prob).mean(dim=-1)
    assert torch.allclose(details["estimate_vec"], expected_vec)
    assert torch.allclose(
        details["estimate_by_pair"],
        expected_vec.mean(dim=0),
    )
    assert torch.allclose(loss, -expected_vec.mean())


def test_sampled_pairwise_ba_default_modules_and_gradients() -> None:
    estimator = SampledPairwiseBA(
        dim=2,
        enc_dim=3,
        sample_size=2,
    )
    batch = PairwiseMIBatch(
        x=torch.randn(8, 4, 2),
        batch_size=[8],
    )

    predictions = estimator.compute_forward(batch)
    objective = estimator.compute_objectives(predictions)
    objective.loss.backward()

    assert isinstance(estimator.encoder, nn.Linear)
    assert predictions.hx.shape == (8, 2, 2, 3)
    assert estimator.encoder.weight.grad is not None
    assert estimator.conditional_proposal.mu.weight.grad is not None


def test_sampled_pairwise_ba_validates_sample_size_and_masks() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        SampledPairwiseBA(dim=2, enc_dim=2, sample_size=0)

    estimator = make_sampled_pairwise_ba(sample_size=2)
    empty = PairwiseMIBatch(
        x=torch.randn(2, 4, 2),
        x_mask=torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]),
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="at least two valid"):
        estimator.compute_forward(empty)

    invalid_mask = PairwiseMIBatch(
        x=torch.randn(2, 4, 2),
        x_mask=torch.ones(2, 3, dtype=torch.bool),
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="x_mask"):
        estimator.compute_forward(invalid_mask)


def test_sampled_pairwise_ba_full_mode_returns_per_observation_matrices() -> None:
    estimator = make_sampled_pairwise_ba(sample_size=2)
    mask = torch.tensor(
        [[1, 1, 1, 0], [1, 0, 1, 1]],
        dtype=torch.bool,
    )
    batch = PairwiseMIBatch(
        x=torch.randn(2, 4, 2),
        x_mask=mask.unsqueeze(-1),
        batch_size=[2],
    )

    estimate = estimator.estimate(batch, {"mode": "full"})

    expected = torch.full((2, 4, 4), torch.log(torch.tensor(2.0)))
    pair_valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    expected.masked_fill_(~pair_valid, 0)
    expected.diagonal(dim1=-2, dim2=-1).zero_()
    assert estimate.batch_size == torch.Size([2])
    assert estimate.value.shape == (2, 4, 4)
    assert estimate.entropies.shape == (2, 4)
    assert torch.allclose(estimate.value, expected)
    assert torch.equal(estimate.value, estimate.value.transpose(-1, -2))
    assert torch.count_nonzero(
        estimate.value.diagonal(dim1=-2, dim2=-1)
    ) == 0
    assert torch.equal(estimate.details["mask"], mask)
    expected_entropies = torch.full(
        (2, 4),
        torch.log(torch.tensor(4.0)),
    )
    expected_entropies.masked_fill_(~mask, 0)
    assert torch.allclose(estimate.entropies, expected_entropies)


def test_sampled_pairwise_ba_estimate_modes_and_options() -> None:
    estimator = make_sampled_pairwise_ba(sample_size=2)
    batch = PairwiseMIBatch(
        x=torch.randn(3, 4, 2),
        batch_size=[3],
    )

    default = estimator.estimate(batch)
    sampled = estimator.estimate(batch, {"mode": "sampled"})

    assert default.value.ndim == 0
    assert sampled.value.ndim == 0
    assert default.entropies.shape == (3, 2, 2)
    assert sampled.entropies.shape == (3, 2, 2)
    with pytest.raises(ValueError, match="mode"):
        estimator.estimate(batch, {"mode": "mystery"})
    with pytest.raises(ValueError, match="unknown"):
        estimator.estimate(batch, {"mystery": True})


def test_sampled_pairwise_ba_full_mode_requires_three_dimensions() -> None:
    estimator = make_sampled_pairwise_ba(sample_size=2)
    batch = PairwiseMIBatch(
        x=torch.randn(2, 3, 4, 2),
        batch_size=[2, 3],
    )

    with pytest.raises(ValueError, match=r"\(batch, count, dim\)"):
        estimator.estimate(batch, {"mode": "full"})


def test_sampled_pairwise_ba_full_mode_chunks_pairwise_features() -> None:
    proposal = RecordingDensityProposal(dim=2, density=0.5)
    estimator = SampledPairwiseBA(
        dim=2,
        enc_dim=2,
        sample_size=2,
        encoder=nn.Identity(),
        conditional_proposal=proposal,
        entropy_estimator=ConstantEntropyEstimator(
            2,
            entropy=torch.log(torch.tensor(4.0)).item(),
        ),
    )
    batch = PairwiseMIBatch(
        x=torch.randn(2, 7, 2),
        batch_size=[2],
    )

    chunked = estimator.estimate(
        batch,
        {"mode": "full", "chunk_size": 3},
    )
    unchunked = estimator.estimate(
        batch,
        {"mode": "full", "chunk_size": 7},
    )

    assert torch.equal(chunked.value, unchunked.value)
    chunked_shapes = proposal.condition_shapes[:-2]
    assert len(chunked_shapes) > 2
    assert all(shape[-3] <= 3 for shape in chunked_shapes)
    assert all(shape[-2] <= 3 for shape in chunked_shapes)
    assert all(shape[-1] == 2 for shape in chunked_shapes)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_sampled_pairwise_ba_full_mode_validates_chunk_size(
    chunk_size: object,
) -> None:
    estimator = make_sampled_pairwise_ba(sample_size=2)
    batch = PairwiseMIBatch(
        x=torch.randn(2, 4, 2),
        batch_size=[2],
    )

    with pytest.raises(ValueError, match="chunk_size"):
        estimator.estimate(
            batch,
            {"mode": "full", "chunk_size": chunk_size},
        )

    with pytest.raises(ValueError, match="only supported in full mode"):
        estimator.estimate(batch, {"chunk_size": 2})
