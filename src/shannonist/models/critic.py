from collections.abc import Sequence

import torch
from tensordict import TensorClass
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from shannonist.models.mlp import MLP


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
