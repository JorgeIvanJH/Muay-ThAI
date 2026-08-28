"""Fixed-size temporal history used by live action classifiers."""

from __future__ import annotations

import numpy as np


class TemporalWindowBuffer:
    """Maintain only the current left-padded causal feature window."""

    def __init__(self, window_size: int, feature_count: int) -> None:
        """Allocate a reusable circular buffer and model input window.

        Usage: Inference only.
        """

        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if feature_count < 1:
            raise ValueError("feature_count must be at least 1")
        self.window_size = int(window_size)
        self.feature_count = int(feature_count)
        self._ring = np.zeros(
            (self.window_size, self.feature_count),
            dtype=np.float32,
        )
        self._window = np.zeros_like(self._ring)
        self._count = 0
        self._next_index = 0

    def append(self, features: np.ndarray) -> None:
        """Append one feature vector, overwriting the oldest when full.

        Usage: Inference only.
        """

        values = np.asarray(features, dtype=np.float32)
        if values.shape != (self.feature_count,):
            raise ValueError(
                f"features must have shape ({self.feature_count},), got "
                f"{values.shape}"
            )
        self._ring[self._next_index] = values
        self._next_index = (self._next_index + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)

    def current_window(self) -> np.ndarray:
        """Return the current chronological, left-zero-padded window.

        Usage: Inference only.

        The returned array is reused by the buffer and should be consumed
        before the next call to ``append``.
        """

        self._window.fill(0.0)
        if self._count == 0:
            return self._window
        if self._count < self.window_size:
            self._window[-self._count :] = self._ring[: self._count]
            return self._window

        first_count = self.window_size - self._next_index
        self._window[:first_count] = self._ring[self._next_index :]
        if self._next_index:
            self._window[first_count:] = self._ring[: self._next_index]
        return self._window
