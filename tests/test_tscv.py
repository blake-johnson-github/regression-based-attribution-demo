from __future__ import annotations

import numpy as np
import pytest

from attrib_regression.eval.tscv import TimeSeriesCV


def test_basic_split():
    cv = TimeSeriesCV(n_splits=3, test_size=2)
    folds = list(cv.split(10))
    assert len(folds) == 3
    for train, test in folds:
        assert len(test) == 2
        # train and test don't overlap
        assert len(np.intersect1d(train, test)) == 0


def test_expanding_window():
    """Each successive fold should have more training data."""
    cv = TimeSeriesCV(n_splits=3, test_size=2)
    folds = list(cv.split(12))
    train_sizes = [len(tr) for tr, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]


def test_gap_reduces_training():
    cv_no_gap = TimeSeriesCV(n_splits=2, test_size=3, gap=0)
    cv_gap = TimeSeriesCV(n_splits=2, test_size=3, gap=2)
    folds_no = list(cv_no_gap.split(20))
    folds_gap = list(cv_gap.split(20))
    # with gap, training set is smaller
    for (tr_no, _), (tr_gap, _) in zip(folds_no, folds_gap):
        assert len(tr_gap) < len(tr_no)


def test_test_indices_cover_end():
    cv = TimeSeriesCV(n_splits=2, test_size=3)
    folds = list(cv.split(10))
    # last fold's test should end at n_samples
    _, last_test = folds[-1]
    assert last_test[-1] == 9


def test_too_few_samples_raises():
    cv = TimeSeriesCV(n_splits=5, test_size=10)
    with pytest.raises(ValueError, match="Not enough samples"):
        list(cv.split(50))


def test_empty_training_from_gap_raises():
    cv = TimeSeriesCV(n_splits=3, test_size=2, gap=100)
    with pytest.raises(ValueError, match="no training samples"):
        list(cv.split(10))


def test_invalid_n_splits():
    cv = TimeSeriesCV(n_splits=0, test_size=2)
    with pytest.raises(ValueError, match="n_splits"):
        list(cv.split(10))


def test_invalid_test_size():
    cv = TimeSeriesCV(n_splits=2, test_size=0)
    with pytest.raises(ValueError, match="test_size"):
        list(cv.split(10))
