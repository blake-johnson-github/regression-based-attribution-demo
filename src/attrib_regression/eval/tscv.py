from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class TimeSeriesCV:
    n_splits: int
    test_size: int
    gap: int = 0

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Expanding window splits with fixed test size."""
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")

        total_test = self.n_splits * self.test_size
        if n_samples <= total_test:
            raise ValueError("Not enough samples for the requested CV splits.")

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size
            train_end = max(0, test_start - self.gap)
            if train_end < 1:
                raise ValueError(
                    f"Fold {i}: gap={self.gap} leaves no training samples. "
                    f"Reduce gap or n_splits."
                )
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx
