"""Activation functions used inside transformer feed-forward networks."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


ActivationFunction = Callable[
    [NDArray[np.floating]],
    NDArray[np.floating],
]


def relu(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the rectified linear activation of every input value."""
    return np.maximum(values, 0.0)


def gelu_new(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return GPT-2's tanh approximation of the GELU activation."""
    coefficient = np.sqrt(2.0 / np.pi)

    return 0.5 * values * (
        1.0
        + np.tanh(
            coefficient * (values + 0.044715 * values**3)
        )
    )
