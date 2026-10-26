import numpy as np
import pytest
from scipy.special import gammaln

from efsc.preprocessing import (
    bernoulli_sufficient_statistics_flat,
    chi_square_sufficient_statistics_flat,
    exponential_sufficient_statistics_flat,
    gaussian_known_sigma_sufficient_statistics_flat,
    gaussian_sufficient_statistics_flat,
    laplace_sufficient_statistics_flat,
    multinomial_sufficient_statistics_flat,
    poisson_sufficient_statistics_flat,
)


def _ragged_panel(values):
    panel = np.empty((2, 2), dtype=object)
    panel[0, 0] = np.asarray(values[0], dtype=float)
    panel[0, 1] = np.asarray(values[1], dtype=float)
    panel[1, 0] = np.asarray(values[2], dtype=float)
    panel[1, 1] = np.asarray(values[3], dtype=float)
    return panel


@pytest.mark.parametrize(
    ("preprocessor", "values", "kwargs", "statistic", "carrier"),
    [
        (
            poisson_sufficient_statistics_flat,
            [[0, 2], [9], [1], [3, 4, 5]],
            {},
            lambda y: np.sum(y),
            lambda y: -np.sum(gammaln(1 + y)),
        ),
        (
            bernoulli_sufficient_statistics_flat,
            [[0, 1], [1], [1, 1], [0, 0, 1]],
            {},
            lambda y: np.sum(y),
            lambda y: 0.0,
        ),
        (
            exponential_sufficient_statistics_flat,
            [[0.5, 2], [9], [1.5], [3, 4, 5]],
            {},
            lambda y: np.sum(y),
            lambda y: 0.0,
        ),
        (
            laplace_sufficient_statistics_flat,
            [[-1, 2], [9], [1.5], [3, 4, 5]],
            {"mu": 1.0},
            lambda y: np.sum(np.abs(y - 1.0)),
            lambda y: 0.0,
        ),
        (
            chi_square_sufficient_statistics_flat,
            [[0.5, 2], [9], [1.5], [3, 4, 5]],
            {},
            lambda y: np.sum(np.log(y)),
            lambda y: -np.sum(y / 2),
        ),
        (
            gaussian_known_sigma_sufficient_statistics_flat,
            [[-1, 2], [9], [1.5], [3, 4, 5]],
            {"sigma": 2.0},
            lambda y: np.sum(y / 2.0),
            lambda y: -0.5
            * np.sum((y / 2.0) ** 2 + np.log(2 * np.pi * 2.0**2)),
        ),
    ],
)
def test_one_parameter_ragged_sufficient_statistics(
    preprocessor,
    values,
    kwargs,
    statistic,
    carrier,
):
    panel = _ragged_panel(values)
    mask = np.array([[1, 0], [1, 1]], dtype=bool)

    unit_idx, time_idx, m_vec, tx_vec, carrier_vec = preprocessor(
        y=panel,
        mask=mask,
        device="cpu",
        **kwargs,
    )

    selected = [panel[0, 0], panel[1, 0], panel[1, 1]]
    np.testing.assert_array_equal(unit_idx.numpy(), [0, 1, 1])
    np.testing.assert_array_equal(time_idx.numpy(), [0, 0, 1])
    np.testing.assert_allclose(m_vec.numpy(), [len(y) for y in selected])
    np.testing.assert_allclose(tx_vec.numpy(), [statistic(y) for y in selected])
    np.testing.assert_allclose(
        carrier_vec.numpy(),
        [carrier(y) for y in selected],
        rtol=1e-6,
    )


def test_gaussian_ragged_sufficient_statistics():
    panel = _ragged_panel([[1, 2], [9], [-1], [3, 4, 5]])
    mask = np.array([[1, 0], [1, 1]], dtype=bool)

    unit_idx, time_idx, m_vec, tx0, tx1, log_carrier = (
        gaussian_sufficient_statistics_flat(panel, mask)
    )

    np.testing.assert_array_equal(unit_idx.numpy(), [0, 1, 1])
    np.testing.assert_array_equal(time_idx.numpy(), [0, 0, 1])
    np.testing.assert_allclose(m_vec.numpy(), [2, 1, 3])
    np.testing.assert_allclose(tx0.numpy(), [3, -1, 12])
    np.testing.assert_allclose(tx1.numpy(), [5, 1, 50])
    np.testing.assert_allclose(
        log_carrier.numpy(),
        np.log(1 / np.sqrt(2 * np.pi)),
    )


def test_multinomial_sufficient_statistics_and_reference_category():
    counts = np.array(
        [
            [[2, 3, 5], [1, 0, 2]],
            [[0, 4, 1], [3, 3, 4]],
        ]
    )
    mask = np.array([[1, 0], [1, 1]], dtype=bool)

    unit_idx, time_idx, m_vec, tx_vec, carrier_vec = (
        multinomial_sufficient_statistics_flat(counts, mask)
    )

    selected = counts[mask]
    np.testing.assert_array_equal(unit_idx.numpy(), [0, 1, 1])
    np.testing.assert_array_equal(time_idx.numpy(), [0, 0, 1])
    np.testing.assert_allclose(m_vec.numpy(), selected.sum(axis=1))
    np.testing.assert_allclose(tx_vec.numpy(), selected[:, :-1])
    np.testing.assert_allclose(
        carrier_vec.numpy(),
        gammaln(1 + selected.sum(axis=1)) - gammaln(1 + selected).sum(axis=1),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "counts",
    [
        np.array([[[-1, 2]]]),
        np.array([[[0.5, 1.5]]]),
    ],
)
def test_multinomial_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        multinomial_sufficient_statistics_flat(counts, np.ones((1, 1)))
