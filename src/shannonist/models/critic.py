from collections.abc import Sequence

import torch
from tensordict import TensorClass
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from shannonist.models.mlp import MLP, MultiMLP


class BilinearPotential(nn.Module):
    """MLP potential operating on a pair of feature representations.

    The two inputs are concatenated along their feature dimension and passed
    to an internally constructed :class:`MLP`. Consequently, ``input_dim``
    describes the width of each individual input, while the internal MLP has
    an input width of ``2 * input_dim``.

    Parameters
    ----------
    input_dim : int
        Feature dimension of each input representation.
    output_dim : int, default=1
        Number of output features.
    hidden_dim : Sequence[int], default=(512, 512)
        Width of each hidden layer.
    act_func : nn.Module, optional
        Activation applied after each hidden layer. Defaults to ``nn.ReLU``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: Sequence[int] = (512, 512),
        act_func: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=input_dim * 2,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            act_func=act_func,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the potential for two feature representations.

        Parameters
        ----------
        x : Tensor
            First representation with shape ``(batch, input_dim)``.
        y : Tensor
            Second representation with shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            Potential values with shape ``(batch, output_dim)``.
        """
        return self.mlp(torch.cat((x, y), dim=1))


class PairwiseCritic(nn.Module):
    r"""Compute symmetric pairwise interactions between encoded inputs.

    A :class:`MultiMLP` independently encodes every position in the count
    dimension. Pairwise scores are then computed as

    .. math::

        s_{ij} = h_i^\mathsf{T} W h_j, \qquad W = A^\mathsf{T} A.

    The factorization makes ``W`` symmetric and positive semidefinite, so
    swapping ``i`` and ``j`` does not change the interaction score.

    Parameters
    ----------
    encoder : MultiMLP
        Parallel encoder mapping ``(*, count, input_features)`` to
        ``(*, count, feature_dim)``.
    count : int
        Size expected in the input's penultimate dimension.
    use_norm : bool, default=True
        Whether to L2-normalize encoded representations along their feature
        dimension.

    Attributes
    ----------
    A : nn.Parameter
        Learned matrix whose Gram matrix defines the symmetric interaction.
    """

    def __init__(
        self,
        encoder: MultiMLP,
        count: int,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if count <= 0:
            raise ValueError("count must be positive")
        if encoder.count != count:
            raise ValueError(
                f"encoder count {encoder.count} does not match critic count {count}"
            )

        self.encoder = encoder
        self.count = count
        self.feature_dim = encoder.output_dim
        self.use_norm = use_norm
        self.A = nn.Parameter(torch.empty(self.feature_dim, self.feature_dim))
        nn.init.xavier_uniform_(self.A)

    @property
    def weight(self) -> Tensor:
        """Return the symmetric positive-semidefinite interaction matrix."""
        return self.A.transpose(0, 1) @ self.A

    def forward(self, x: Tensor) -> Tensor:
        """Compute every pairwise interaction score.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, count, input_features)``.

        Returns
        -------
        Tensor
            Symmetric interaction matrix with shape ``(*, count, count)``.

        Raises
        ------
        ValueError
            If the input or encoded representation has an incompatible shape.
        """
        hx = self.encode(x)
        return self.compute_interactions(hx)

    def encode(self, x: Tensor) -> Tensor:
        """Encode and optionally normalize all count positions.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, count, input_features)``.

        Returns
        -------
        Tensor
            Encoded input with shape ``(*, count, feature_dim)``.
        """
        if x.ndim < 2:
            raise ValueError("input must have shape (*, count, features)")
        if x.shape[-2] != self.count:
            raise ValueError(
                f"expected count dimension {self.count}, got {x.shape[-2]}"
            )

        hx = self.encoder(x)
        if hx.shape[-2] != self.count or hx.shape[-1] != self.feature_dim:
            raise ValueError(
                "encoder must return shape (*, count, encoder.output_dim)"
            )
        if self.use_norm:
            hx = F.normalize(hx, dim=-1)
        return hx

    def compute_interactions(self, hx: Tensor) -> Tensor:
        """Compute symmetric interactions from encoded representations.

        Parameters
        ----------
        hx : Tensor
            Encoded representations with shape
            ``(*, count, feature_dim)``.

        Returns
        -------
        Tensor
            Symmetric interaction matrix with shape ``(*, count, count)``.
        """
        if hx.ndim < 2:
            raise ValueError("encoded input must have shape (*, count, feature_dim)")
        if hx.shape[-2:] != (self.count, self.feature_dim):
            raise ValueError(
                "encoded input must have shape (*, count, feature_dim)"
            )

        interaction = (hx @ self.weight) @ hx.transpose(-2, -1)
        return (interaction + interaction.transpose(-2, -1)) / 2


class SymmetricPairwiseCritic(nn.Module):
    r"""Compute symmetric pairwise interactions with a shared encoder.

    A single :class:`MLP` encodes every item in the input's penultimate
    dimension using shared parameters. Pairwise scores are then computed as

    .. math::

        s_{ij} = h_i^\mathsf{T} W h_j, \qquad W = A^\mathsf{T} A.

    The factorization makes ``W`` symmetric and positive semidefinite. Unlike
    :class:`PairwiseCritic`, this critic has no fixed count: the number of
    items may vary between calls.

    Parameters
    ----------
    encoder : MLP
        Shared encoder mapping each input feature vector to ``output_dim``.
    use_norm : bool, default=True
        Whether to L2-normalize encoded representations along their feature
        dimension.

    Attributes
    ----------
    A : nn.Parameter
        Learned matrix whose Gram matrix defines the symmetric interaction.
    """

    def __init__(self, encoder: MLP, use_norm: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = encoder.output_dim
        self.use_norm = use_norm
        self.A = nn.Parameter(torch.empty(self.feature_dim, self.feature_dim))
        nn.init.xavier_uniform_(self.A)

    @property
    def weight(self) -> Tensor:
        """Return the symmetric positive-semidefinite interaction matrix."""
        return self.A.transpose(0, 1) @ self.A

    def forward(self, x: Tensor) -> Tensor:
        """Compute every pairwise interaction score.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, count, input_features)``. ``count`` may
            have any positive value.

        Returns
        -------
        Tensor
            Symmetric interaction matrix with shape ``(*, count, count)``.
        """
        return self.compute_interactions(self.encode(x))

    def encode(self, x: Tensor) -> Tensor:
        """Encode all items using the same MLP and optionally normalize them.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, count, input_features)``.

        Returns
        -------
        Tensor
            Encoded input with shape ``(*, count, feature_dim)``.

        Raises
        ------
        ValueError
            If the input does not contain count and feature dimensions or its
            feature dimension is incompatible with the encoder.
        """
        if x.ndim < 2:
            raise ValueError("input must have shape (*, count, features)")
        if x.shape[-2] <= 0:
            raise ValueError("count dimension must be positive")
        if x.shape[-1] != self.encoder.input_dim:
            raise ValueError(
                f"expected feature dimension {self.encoder.input_dim}, "
                f"got {x.shape[-1]}"
            )

        flattened = x.reshape(-1, x.shape[-1])
        hx = self.encoder(flattened).reshape(*x.shape[:-1], self.feature_dim)
        if self.use_norm:
            hx = F.normalize(hx, dim=-1)
        return hx

    def compute_interactions(self, hx: Tensor) -> Tensor:
        """Compute symmetric interactions from encoded representations.

        Parameters
        ----------
        hx : Tensor
            Encoded representations with shape
            ``(*, count, feature_dim)``.

        Returns
        -------
        Tensor
            Symmetric interaction matrix with shape ``(*, count, count)``.

        Raises
        ------
        ValueError
            If the encoded representation has an incompatible shape.
        """
        if hx.ndim < 2:
            raise ValueError("encoded input must have shape (*, count, feature_dim)")
        if hx.shape[-2] <= 0 or hx.shape[-1] != self.feature_dim:
            raise ValueError(
                "encoded input must have shape (*, count, feature_dim)"
            )

        interaction = (hx @ self.weight) @ hx.transpose(-2, -1)
        return (interaction + interaction.transpose(-2, -1)) / 2


class BilinearCriticOutput(TensorClass):
    """Output produced by :class:`BilinearCritic`.

    Parameters
    ----------
    hx : Tensor
        Temperature-scaled representation of the first input.
    hy : Tensor
        Temperature-scaled representation of the second input.
    u : Tensor
        Interaction computed from the unscaled representations.
    """

    hx: Tensor
    hy: Tensor
    u: Tensor


class BilinearCritic(nn.Module):
    """Encode two inputs for use in a bilinear critic.

    This critic follows the formulation in https://arxiv.org/abs/2107.01131.
    The two encoders must map their respective inputs to the same feature
    dimension. ``potential`` receives both representations and computes their
    interaction term.

    Parameters
    ----------
    encoder_x : nn.Module
        Module mapping the first input to a feature representation.
    encoder_y : nn.Module
        Module mapping the second input to a feature representation.
    potential : nn.Module
        Module accepting both normalized feature representations.
    tau : float, default=1.0
        Initial value of the learnable temperature parameter.
    use_norm : bool, default=True
        Whether to L2-normalize the encoded representations.
    """

    def __init__(
        self,
        encoder_x: nn.Module,
        encoder_y: nn.Module,
        potential: nn.Module,
        tau: float = 1.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.potential = potential
        self.tau = nn.Parameter(torch.as_tensor([tau]))
        self.use_norm = use_norm

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        tau: Tensor | None = None,
    ) -> BilinearCriticOutput:
        """Compute temperature-scaled representations and their interaction.

        Parameters
        ----------
        x : Tensor
            Input tensor for ``encoder_x``.
        y : Tensor
            Input tensor for ``encoder_y``.
        tau : Tensor, optional
            Temperature override. The module's learnable temperature is used
            when this is omitted.

        Returns
        -------
        BilinearCriticOutput
            Structured output containing the scaled representations and the
            interaction computed by ``potential``.
        """
        if tau is None:
            tau = self.tau
        tau = torch.sqrt(tau)
        hx = self.encoder_x(x)
        hy = self.encoder_y(y)
        if self.use_norm:
            hx = self.norm(hx)
            hy = self.norm(hy)
        u = self.potential(hx, hy)

        return BilinearCriticOutput(
            hx=hx / tau,
            hy=hy / tau,
            u=u,
            batch_size=hx.shape[:1],
        )

    @staticmethod
    def norm(z: Tensor) -> Tensor:
        """L2-normalize feature vectors along their feature axis.

        Parameters
        ----------
        z : Tensor
            Batch of feature vectors.

        Returns
        -------
        Tensor
            Feature vectors normalized along dimension 1.
        """
        return F.normalize(z, dim=1)
