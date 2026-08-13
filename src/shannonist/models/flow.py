"""Interfaces for invertible neural-network transformations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
import math

import torch
from tensordict import TensorClass
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, Normal
from torch.nn import functional as F


class InvertibleOutput(TensorClass):
    """Output of an invertible transformation.

    Parameters
    ----------
    value : Tensor
        Transformed tensor.
    log_abs_det : Tensor
        Logarithm of the absolute Jacobian determinant for each transformed
        value.
    """

    value: Tensor
    log_abs_det: Tensor


class FlowDensityOutput(TensorClass):
    """Sample or density evaluation produced by a normalizing flow.

    Parameters
    ----------
    value : Tensor
        Value in the modeled data space.
    latent : Tensor
        Corresponding value in the prior's latent space.
    log_prob : Tensor
        Log-density of ``value`` under the flow distribution.
    log_abs_det : Tensor
        Log-absolute-determinant used for this transformation direction.
    """

    value: Tensor
    latent: Tensor
    log_prob: Tensor
    log_abs_det: Tensor


class Invertible(nn.Module, ABC):
    """Interface for an invertible neural-network transformation."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply the forward transformation.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        cond : Tensor, optional
            Optional conditioning tensor. Implementations that do not support
            conditioning may ignore it.

        Returns
        -------
        InvertibleOutput
            Transformed tensor and forward log-absolute-determinant.
        """
        ...

    @abstractmethod
    def inverse(
        self,
        y: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply the inverse transformation.

        Parameters
        ----------
        y : Tensor
            Transformed tensor.
        cond : Tensor, optional
            Optional conditioning tensor. Implementations that do not support
            conditioning may ignore it.

        Returns
        -------
        InvertibleOutput
            Reconstructed input and inverse log-absolute-determinant.
        """
        ...


class InvertibleLinear(Invertible):
    r"""Apply an invertible affine transformation.

    The forward transformation has the same convention as
    :class:`torch.nn.Linear`:

    .. math::

        y = x W^\mathsf{T} + b.

    The square weight is parameterized as ``P @ L @ U``. Initialization first
    samples an orthogonal matrix and decomposes it into PLU factors. ``P`` and
    the signs of the diagonal of ``U`` remain fixed, while the strict
    triangles and logarithm of the diagonal magnitudes are learned. The
    resulting matrix is therefore invertible by construction. :meth:`inverse`
    uses triangular solves rather than explicitly constructing a matrix
    inverse.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    bias : bool, default=True
        Whether to include an additive bias.
    gain : float, default=1.0
        Scale applied to the initial orthogonal weight matrix.

    Attributes
    ----------
    l_params : nn.Parameter
        Unconstrained parameters for the strict lower triangle of ``L``.
    u_params : nn.Parameter
        Unconstrained parameters for the strict upper triangle of ``U``.
    log_diag : nn.Parameter
        Logarithm of the absolute diagonal of ``U``.
    P : Tensor
        Fixed permutation matrix.
    sign_diag : Tensor
        Fixed signs of the diagonal of ``U``.
    weight : Tensor
        Assembled square transformation matrix with shape ``(dim, dim)``.
    bias : nn.Parameter or None
        Optional bias with shape ``(dim,)``.
    """

    P: Tensor
    sign_diag: Tensor

    def __init__(
        self,
        dim: int,
        bias: bool = True,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive")
        self.dim = dim
        self.gain = gain
        self.l_params = nn.Parameter(torch.empty(dim, dim))
        self.u_params = nn.Parameter(torch.empty(dim, dim))
        self.log_diag = nn.Parameter(torch.empty(dim))
        self.register_buffer("P", torch.empty(dim, dim))
        self.register_buffer("sign_diag", torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize from the PLU decomposition of an orthogonal matrix."""
        with torch.no_grad():
            initial_weight = torch.empty_like(self.l_params)
            nn.init.orthogonal_(initial_weight, gain=self.gain)
            lu, pivots = torch.linalg.lu_factor(initial_weight)
            permutation, lower, upper = torch.lu_unpack(lu, pivots)
            diagonal = upper.diagonal()

            self.P.copy_(permutation)
            self.sign_diag.copy_(diagonal.sign())
            self.l_params.copy_(torch.tril(lower, diagonal=-1))
            self.u_params.copy_(torch.triu(upper, diagonal=1))
            self.log_diag.copy_(diagonal.abs().log())
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def L(self) -> Tensor:
        """Return the unit lower-triangular factor."""
        identity = torch.eye(
            self.dim,
            dtype=self.l_params.dtype,
            device=self.l_params.device,
        )
        return torch.tril(self.l_params, diagonal=-1) + identity

    def U(self) -> Tensor:
        """Return the upper-triangular factor with nonzero diagonal."""
        return torch.triu(self.u_params, diagonal=1) + torch.diag(
            self.sign_diag * self.log_diag.exp()
        )

    @property
    def weight(self) -> Tensor:
        """Return the assembled invertible weight matrix."""
        return self.P @ self.L() @ self.U()

    @property
    def log_abs_det(self) -> Tensor:
        """Return the log-absolute-determinant of the weight."""
        return self.log_diag.sum()

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply the affine transformation.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, dim)``.
        cond : Tensor, optional
            Ignored; this transformation is unconditional.

        Returns
        -------
        InvertibleOutput
            Transformed values and forward log-absolute-determinants with
            shapes ``(*, dim)`` and ``(*)``, respectively.

        Raises
        ------
        ValueError
            If the input feature dimension is incompatible.
        """
        self._validate_input(x)
        batch_size = x.shape[:-1]
        return InvertibleOutput(
            value=F.linear(x, self.weight, self.bias),
            log_abs_det=self.log_abs_det.expand(batch_size),
            batch_size=batch_size,
        )

    def inverse(
        self,
        y: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Invert the affine transformation using an LU solve.

        Parameters
        ----------
        y : Tensor
            Transformed values with shape ``(*, dim)``.
        cond : Tensor, optional
            Ignored; this transformation is unconditional.

        Returns
        -------
        InvertibleOutput
            Reconstructed inputs and inverse log-absolute-determinants with
            shapes ``(*, dim)`` and ``(*)``, respectively.

        Raises
        ------
        ValueError
            If the input feature dimension is incompatible.
        """
        self._validate_input(y)
        centered = y if self.bias is None else y - self.bias
        flattened = centered.reshape(-1, self.dim)
        right_hand_side = self.P.transpose(0, 1) @ flattened.transpose(0, 1)
        intermediate = torch.linalg.solve_triangular(
            self.L(),
            right_hand_side,
            upper=False,
            unitriangular=True,
        )
        solution = torch.linalg.solve_triangular(
            self.U(),
            intermediate,
            upper=True,
        )
        batch_size = y.shape[:-1]
        return InvertibleOutput(
            value=solution.transpose(0, 1).reshape_as(centered),
            log_abs_det=(-self.log_abs_det).expand(batch_size),
            batch_size=batch_size,
        )

    def _validate_input(self, x: Tensor) -> None:
        """Validate the trailing feature dimension."""
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"input must have trailing dimension {self.dim}")


class ConditionedInvertibleLinearLayer(Invertible):
    r"""Apply an affine map parameterized by a conditioning hypernetwork.

    The hypernetwork maps ``cond`` to the strict triangles of ``L`` and ``U``,
    the logarithm of the positive diagonal of ``U``, and an optional bias. The
    resulting weight ``L @ U`` is invertible by construction for every
    conditioning value. The hypernetwork's output layer is initialized to zero,
    making the initial transformation the identity map.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    cond_dim : int
        Trailing dimension of the conditioning tensor.
    hidden_dims : Sequence[int], optional
        Widths of the hypernetwork hidden layers. Defaults to one layer of
        width ``dim``.
    bias : bool, default=True
        Whether the hypernetwork should produce an additive bias.
    """

    lower_indices: Tensor
    upper_indices: Tensor

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        hidden_dims: Sequence[int] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if cond_dim <= 0:
            raise ValueError("cond_dim must be positive")
        if hidden_dims is None:
            hidden_dims = (dim,)
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("all hidden dimensions must be positive")

        self.dim = dim
        self.cond_dim = cond_dim
        self.hidden_dims = tuple(hidden_dims)
        self.use_bias = bias
        self.triangle_size = dim * (dim - 1) // 2
        self.register_buffer(
            "lower_indices",
            torch.tril_indices(dim, dim, offset=-1),
        )
        self.register_buffer(
            "upper_indices",
            torch.triu_indices(dim, dim, offset=1),
        )
        parameter_dim = 2 * self.triangle_size + dim
        if bias:
            parameter_dim += dim

        widths = (cond_dim, *self.hidden_dims, parameter_dim)
        modules: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(
            zip(widths[:-1], widths[1:], strict=True)
        ):
            linear = nn.Linear(input_width, output_width)
            if index < len(widths) - 2:
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                modules.extend((linear, nn.ReLU()))
            else:
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
                modules.append(linear)
        self.hypernetwork = nn.Sequential(*modules)

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply the conditioned affine transformation."""
        x, cond = self._validate_and_broadcast(x, cond)
        lower, upper, bias, log_abs_det = self._factors(cond)
        value = torch.einsum("...ij,...j->...i", upper, x)
        value = torch.einsum("...ij,...j->...i", lower, value)
        if bias is not None:
            value = value + bias
        return InvertibleOutput(
            value=value,
            log_abs_det=log_abs_det,
            batch_size=x.shape[:-1],
        )

    def inverse(
        self,
        y: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Invert the conditioned affine transformation."""
        y, cond = self._validate_and_broadcast(y, cond)
        lower, upper, bias, log_abs_det = self._factors(cond)
        centered = y if bias is None else y - bias
        intermediate = torch.linalg.solve_triangular(
            lower,
            centered.unsqueeze(-1),
            upper=False,
            unitriangular=True,
        )
        value = torch.linalg.solve_triangular(
            upper,
            intermediate,
            upper=True,
        ).squeeze(-1)
        return InvertibleOutput(
            value=value,
            log_abs_det=-log_abs_det,
            batch_size=y.shape[:-1],
        )

    def _factors(
        self,
        cond: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        """Construct batched triangular factors from the hypernetwork."""
        parameters = self.hypernetwork(cond)
        offset = 0
        lower_params = parameters[..., : self.triangle_size]
        offset += self.triangle_size
        upper_params = parameters[..., offset : offset + self.triangle_size]
        offset += self.triangle_size
        log_diag = parameters[..., offset : offset + self.dim]
        offset += self.dim
        bias = parameters[..., offset:] if self.use_bias else None

        shape = (*cond.shape[:-1], self.dim, self.dim)
        lower = cond.new_zeros(shape)
        upper = cond.new_zeros(shape)
        lower[..., self.lower_indices[0], self.lower_indices[1]] = lower_params
        upper[..., self.upper_indices[0], self.upper_indices[1]] = upper_params
        identity = torch.eye(self.dim, dtype=cond.dtype, device=cond.device)
        lower = lower + identity
        upper = upper + torch.diag_embed(log_diag.exp())
        return lower, upper, bias, log_diag.sum(dim=-1)

    def _validate_and_broadcast(
        self,
        value: Tensor,
        cond: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Validate feature dimensions and broadcast batch dimensions."""
        if value.ndim == 0 or value.shape[-1] != self.dim:
            raise ValueError(
                f"input must have trailing dimension {self.dim}"
            )
        if cond is None:
            raise ValueError("cond is required for a conditioned layer")
        if cond.ndim == 0 or cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond must have trailing dimension {self.cond_dim}"
            )
        batch_shape = torch.broadcast_shapes(
            value.shape[:-1],
            cond.shape[:-1],
        )
        return (
            value.expand(*batch_shape, self.dim),
            cond.expand(*batch_shape, self.cond_dim),
        )


class InvertibleLeakyReLU(Invertible):
    r"""Apply an invertible leaky-ReLU activation.

    For positive inputs the slope is one; for negative inputs it is
    ``negative_slope``. Requiring a strictly positive negative slope makes the
    transformation bijective.

    Parameters
    ----------
    negative_slope : float, default=0.5
        Slope applied to negative inputs. Must be strictly positive.
    """

    def __init__(self, negative_slope: float = 0.5) -> None:
        super().__init__()
        if not math.isfinite(negative_slope) or negative_slope <= 0:
            raise ValueError("negative_slope must be finite and positive")
        self.negative_slope = negative_slope

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply the activation and compute its forward log-determinant."""
        if x.ndim == 0:
            raise ValueError("input must have a trailing feature dimension")
        value = F.leaky_relu(x, negative_slope=self.negative_slope)
        negative_count = (x < 0).sum(dim=-1)
        log_slope = x.new_tensor(self.negative_slope).log()
        log_abs_det = negative_count.to(x.dtype) * log_slope
        return InvertibleOutput(
            value=value,
            log_abs_det=log_abs_det,
            batch_size=x.shape[:-1],
        )

    def inverse(
        self,
        y: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply the analytic inverse and its inverse log-determinant."""
        if y.ndim == 0:
            raise ValueError("input must have a trailing feature dimension")
        value = torch.where(y >= 0, y, y / self.negative_slope)
        negative_count = (y < 0).sum(dim=-1)
        log_slope = y.new_tensor(self.negative_slope).log()
        log_abs_det = -negative_count.to(y.dtype) * log_slope
        return InvertibleOutput(
            value=value,
            log_abs_det=log_abs_det,
            batch_size=y.shape[:-1],
        )


class InvertibleMLP(Invertible):
    """Stack invertible affine layers and nonlinear activations.

    Because :class:`InvertibleLinear` is square, every requested width must be
    identical. An invertible activation follows every linear layer except the
    last one.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : Sequence[int]
        Feature dimension for each intermediate layer.
    output_dim : int
        Output feature dimension.
    activation : Invertible, optional
        Activation used after every non-final linear layer. Defaults to
        :class:`InvertibleLeakyReLU`. Each position receives an independent
        copy of the supplied module.
    use_conditioning : bool, default=False
        Whether to replace affine layers with
        :class:`ConditionedInvertibleLinearLayer`. Conditioning tensors then
        have trailing dimension ``input_dim``.
    Raises
    ------
    ValueError
        If any dimension is non-positive or the dimensions are not identical.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: Invertible | None = None,
        use_conditioning: bool = False,
    ) -> None:
        super().__init__()
        dimensions = (input_dim, *hidden_dims, output_dim)
        if any(dim <= 0 for dim in dimensions):
            raise ValueError("all dimensions must be positive")
        if any(dim != input_dim for dim in dimensions[1:]):
            raise ValueError(
                "all dimensions must match for an invertible linear stack"
            )

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = output_dim
        self.dim = input_dim
        self.use_conditioning = use_conditioning
        if activation is None:
            activation = InvertibleLeakyReLU()
        if not isinstance(activation, Invertible):
            raise TypeError("activation must implement Invertible")
        gain = (
            nn.init.calculate_gain(
                "leaky_relu",
                activation.negative_slope,
            )
            if isinstance(activation, InvertibleLeakyReLU)
            else 1.0
        )
        layer_count = len(dimensions) - 1
        if use_conditioning:
            self.layers = nn.ModuleList(
                ConditionedInvertibleLinearLayer(
                    input_dim,
                    cond_dim=input_dim,
                )
                for _ in range(layer_count)
            )
        else:
            self.layers = nn.ModuleList(
                InvertibleLinear(
                    input_dim,
                    gain=gain if index < layer_count - 1 else 1.0,
                )
                for index in range(layer_count)
            )
        self.activations = nn.ModuleList(
            deepcopy(activation) for _ in range(len(self.layers) - 1)
        )

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply every invertible layer in order.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, input_dim)``.
        cond : Tensor, optional
            Optional conditioning tensor propagated to every component.

        Returns
        -------
        InvertibleOutput
            Transformed values and accumulated forward
            log-absolute-determinants.
        """
        log_abs_det = x.new_zeros(x.shape[:-1])
        for index, layer in enumerate(self.layers):
            output = layer(x, cond=cond)
            x = output.value
            log_abs_det = log_abs_det + output.log_abs_det
            if index < len(self.activations):
                output = self.activations[index](x, cond=cond)
                x = output.value
                log_abs_det = log_abs_det + output.log_abs_det
        return InvertibleOutput(
            value=x,
            log_abs_det=log_abs_det,
            batch_size=x.shape[:-1],
        )

    def inverse(
        self,
        y: Tensor,
        cond: Tensor | None = None,
    ) -> InvertibleOutput:
        """Apply every layer inverse in reverse order.

        Parameters
        ----------
        y : Tensor
            Transformed values with shape ``(*, output_dim)``.
        cond : Tensor, optional
            Optional conditioning tensor propagated to every component.

        Returns
        -------
        InvertibleOutput
            Reconstructed inputs and accumulated inverse
            log-absolute-determinants.
        """
        log_abs_det = y.new_zeros(y.shape[:-1])
        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            output = layer.inverse(y, cond=cond)
            y = output.value
            log_abs_det = log_abs_det + output.log_abs_det
            if index > 0:
                output = self.activations[index - 1].inverse(y, cond=cond)
                y = output.value
                log_abs_det = log_abs_det + output.log_abs_det
        return InvertibleOutput(
            value=y,
            log_abs_det=log_abs_det,
            batch_size=y.shape[:-1],
        )


class FlowDensityEstimator(nn.Module):
    r"""Density model obtained by transforming a prior distribution.

    If ``z`` follows the configured prior and ``x = f(z)``, the modeled
    density is

    .. math::

        \log p_X(x) = \log p_Z(f^{-1}(x))
        + \log |\det J_{f^{-1}}(x)|.

    Parameters
    ----------
    transform : Invertible
        Invertible transformation mapping prior samples into data space.
    prior : Distribution, optional
        Prior distribution providing ``sample`` and ``log_prob``. Defaults to
        an independent standard Gaussian whose dimension is inferred from the
        transform.
    """

    prior_loc: Tensor | None
    prior_scale: Tensor | None

    def __init__(
        self,
        transform: Invertible,
        prior: Distribution | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(transform, Invertible):
            raise TypeError("transform must be an Invertible")
        if prior is not None and not isinstance(prior, Distribution):
            raise TypeError("prior must be a torch Distribution")

        self.transform = transform
        self._prior = prior
        if prior is None:
            dim = getattr(transform, "dim", None)
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    "transform must expose a positive integer dim when prior "
                    "is omitted"
                )
            self.register_buffer("prior_loc", torch.zeros(dim))
            self.register_buffer("prior_scale", torch.ones(dim))
        else:
            self.register_buffer("prior_loc", None)
            self.register_buffer("prior_scale", None)

    @property
    def prior(self) -> Distribution:
        """Return the configured or default prior distribution."""
        if self._prior is not None:
            return self._prior
        assert self.prior_loc is not None
        assert self.prior_scale is not None
        return Independent(Normal(self.prior_loc, self.prior_scale), 1)

    def forward(
        self,
        sample_shape: torch.Size | Sequence[int] = torch.Size(),
        cond: Tensor | None = None,
    ) -> FlowDensityOutput:
        """Sample from the prior and transform into data space.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int], default=torch.Size()
            Leading shape of the requested samples.
        cond : Tensor, optional
            Optional conditioning tensor passed to the transformation.

        Returns
        -------
        FlowDensityOutput
            Data-space samples, their latent values, log-densities, and
            forward log-absolute-determinants.
        """
        latent = self.prior.sample(torch.Size(sample_shape))
        transformed = self.transform(latent, cond=cond)
        latent = latent.expand_as(transformed.value)
        prior_log_prob = self.prior.log_prob(latent)
        self._validate_log_prob_shapes(prior_log_prob, transformed.log_abs_det)
        return FlowDensityOutput(
            value=transformed.value,
            latent=latent,
            log_prob=prior_log_prob - transformed.log_abs_det,
            log_abs_det=transformed.log_abs_det,
            batch_size=prior_log_prob.shape,
        )

    def evaluate(
        self,
        value: Tensor,
        cond: Tensor | None = None,
    ) -> FlowDensityOutput:
        """Evaluate data-space values using the inverse transformation.

        Parameters
        ----------
        value : Tensor
            Values in the modeled data space.
        cond : Tensor, optional
            Optional conditioning tensor passed to the transformation.

        Returns
        -------
        FlowDensityOutput
            Input values, recovered latent values, log-densities, and inverse
            log-absolute-determinants.
        """
        inverted = self.transform.inverse(value, cond=cond)
        prior_log_prob = self.prior.log_prob(inverted.value)
        self._validate_log_prob_shapes(prior_log_prob, inverted.log_abs_det)
        return FlowDensityOutput(
            value=value,
            latent=inverted.value,
            log_prob=prior_log_prob + inverted.log_abs_det,
            log_abs_det=inverted.log_abs_det,
            batch_size=prior_log_prob.shape,
        )

    def log_prob(
        self,
        value: Tensor,
        cond: Tensor | None = None,
    ) -> Tensor:
        """Return the flow log-density of arbitrary data-space values."""
        return self.evaluate(value, cond=cond).log_prob

    def prob(
        self,
        value: Tensor,
        cond: Tensor | None = None,
    ) -> Tensor:
        """Return the flow density of arbitrary data-space values."""
        return self.log_prob(value, cond=cond).exp()

    @staticmethod
    def _validate_log_prob_shapes(
        prior_log_prob: Tensor,
        log_abs_det: Tensor,
    ) -> None:
        """Validate compatibility of prior density and flow Jacobian."""
        if prior_log_prob.shape != log_abs_det.shape:
            raise ValueError(
                "prior log_prob and transform log_abs_det must have "
                "identical shapes"
            )


__all__ = [
    "ConditionedInvertibleLinearLayer",
    "FlowDensityEstimator",
    "FlowDensityOutput",
    "Invertible",
    "InvertibleLinear",
    "InvertibleLeakyReLU",
    "InvertibleMLP",
    "InvertibleOutput",
]
