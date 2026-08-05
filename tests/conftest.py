import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_torch_seed() -> None:
    """Start every test from a deterministic Torch random state."""
    torch.manual_seed(0)
