import pytest
import torch

from shannonist.models import (
    AttentionPoolingConditioning,
    IdentityConditioning,
    make_conditioning,
)


def test_attention_pooling_conditioning_shapes_and_weights() -> None:
    pooling = AttentionPoolingConditioning(dim=2)
    with torch.no_grad():
        pooling.score.weight.copy_(torch.tensor([[1.0, 0.0]]))
        assert pooling.score.bias is not None
        pooling.score.bias.zero_()
    x = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]],
            [[2.0, 4.0], [0.0, 2.0], [-1.0, 8.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)

    output = pooling(x, mask)

    first_weights = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    second_weights = torch.softmax(torch.tensor([0.0, -1.0]), dim=0)
    expected = torch.stack(
        (
            first_weights @ x[0, :2],
            second_weights @ x[1, 1:],
        )
    )
    assert output.shape == (2, 2)
    assert torch.allclose(output, expected)


def test_attention_pooling_conditioning_ignores_masked_values() -> None:
    pooling = AttentionPoolingConditioning(dim=3)
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]])
    modified = x.clone()
    modified[~mask.bool()] = 1_000_000

    expected = pooling(x, mask.unsqueeze(-1))
    actual = pooling(modified, mask)

    assert torch.allclose(actual, expected)


def test_attention_pooling_conditioning_handles_empty_rows() -> None:
    pooling = AttentionPoolingConditioning(dim=3)
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])

    output = pooling(x, mask)

    assert torch.equal(output[0], torch.zeros(3))
    assert torch.isfinite(output).all()


def test_attention_pooling_conditioning_validates_shapes() -> None:
    with pytest.raises(ValueError, match="dim"):
        AttentionPoolingConditioning(dim=0)

    pooling = AttentionPoolingConditioning(dim=3)
    with pytest.raises(ValueError, match="x must have shape"):
        pooling(torch.randn(4, 3))
    with pytest.raises(ValueError, match="trailing dimension"):
        pooling(torch.randn(2, 4, 2))
    with pytest.raises(ValueError, match="mask must have shape"):
        pooling(torch.randn(2, 4, 3), torch.ones(2, 3))


def test_conditioning_factory_builds_recognized_modules() -> None:
    identity = make_conditioning("identity", dim=3)
    attention = make_conditioning(
        "attention-pooling",
        dim=3,
        opts={"bias": False},
    )

    assert isinstance(identity, IdentityConditioning)
    assert isinstance(attention, AttentionPoolingConditioning)
    assert attention.score.bias is None


def test_conditioning_factory_rejects_unknown_names_and_options() -> None:
    with pytest.raises(ValueError, match="unknown conditioning"):
        make_conditioning("mystery", dim=3)
    with pytest.raises(ValueError, match="does not accept options"):
        make_conditioning("identity", dim=3, opts={"bias": False})
