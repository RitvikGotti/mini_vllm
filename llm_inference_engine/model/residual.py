"""Residual connections for transformer sublayers."""

import numpy as np
from numpy.typing import NDArray


class ResidualConnection:
    """Add a sublayer update to its original hidden-state input."""

    def forward(
        self,
        residual: NDArray[np.floating],
        sublayer_output: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return elementwise residual and sublayer-output addition."""
        self._validate_hidden_states("residual", residual)
        self._validate_hidden_states("sublayer_output", sublayer_output)

        if residual.shape != sublayer_output.shape:
            raise ValueError(
                "residual and sublayer_output must have the same shape."
            )

        return residual + sublayer_output

    @staticmethod
    def _validate_hidden_states(
        name: str,
        hidden_states: NDArray[np.floating],
    ) -> None:
        """Validate one two-dimensional floating-point hidden-state array."""
        if hidden_states.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")

        if not np.issubdtype(hidden_states.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")
