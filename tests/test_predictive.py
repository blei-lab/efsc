import numpy as np
import pandas as pd
import pytest

from efsc import EFSC, InferenceConfig


def _fit_family(family):
    if family == "multinomial":
        values = ["a", "b", "c", "a"]
    elif family == "bernoulli":
        values = [0, 1, 0, 1]
    elif family == "poisson":
        values = [0, 1, 2, 3]
    elif family in {"exponential", "chi_square"}:
        values = [0.5, 1.0, 1.5, 2.0]
    else:
        values = [-0.5, 0.5, 1.0, 2.0]
    frame = pd.DataFrame(
        {
            "unit": ["A", "A", "B", "B"],
            "time": [1, 1, 1, 2],
            "y": values,
        }
    )
    kwargs = {}
    if family == "multinomial":
        kwargs["categories"] = ["a", "b", "c"]
    if family == "gaussian_known_sigma":
        kwargs["sigma"] = 1.5
    if family == "laplace_known_mu":
        kwargs["mu"] = 0.25
    return EFSC(
        family=family,
        rank=1,
        inference_config=InferenceConfig(epochs=1),
        **kwargs,
    ).fit(frame, outcome="y", unit="unit", time="time")


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "gaussian_known_sigma",
        "poisson",
        "bernoulli",
        "exponential",
        "laplace_known_mu",
        "chi_square",
        "multinomial",
    ],
)
def test_predictive_sampling_supports_every_family(family):
    result = _fit_family(family)
    first = result.sample_predictive(n_draws=20)
    second = result.sample_predictive(n_draws=20)

    assert first.columns.tolist() == [
        "unit",
        "time",
        "unit_index",
        "time_index",
        "draw",
        "outcome",
        "observed",
        "role",
    ]
    assert len(first) == 2 * 2 * 20
    pd.testing.assert_frame_equal(first, second)
    assert first["draw"].min() == 1
    assert first["draw"].max() == 20

    outcomes = first["outcome"].to_numpy()
    if family in {"gaussian", "gaussian_known_sigma", "laplace_known_mu"}:
        assert np.isfinite(outcomes.astype(float)).all()
    elif family in {"poisson", "bernoulli"}:
        assert np.equal(outcomes, np.floor(outcomes.astype(float))).all()
        assert np.all(outcomes.astype(float) >= 0)
        if family == "bernoulli":
            assert set(outcomes).issubset({0, 1})
    elif family in {"exponential", "chi_square"}:
        assert np.all(outcomes.astype(float) >= 0)
    else:
        assert set(outcomes).issubset({"a", "b", "c"})


def test_predictive_cell_selector_covers_all_observed_and_missing_cells():
    result = _fit_family("gaussian")

    all_cells = result.sample_predictive(n_draws=3)
    observed = result.sample_predictive(n_draws=3, cells="observed")
    missing = result.sample_predictive(n_draws=3, cells="missing")

    assert len(all_cells) == 12
    assert len(observed) == 9
    assert len(missing) == 3
    assert observed["observed"].all()
    assert not missing["observed"].any()
    assert missing[["unit", "time"]].drop_duplicates().to_records(index=False).tolist() == [
        ("A", 2)
    ]


def test_explicit_predictive_seed_overrides_derived_seed():
    result = _fit_family("gaussian")
    assert result.derived_seeds == {"factorization": 0, "predictive": 1}

    first = result.sample_predictive(n_draws=5, random_state=91)
    second = result.sample_predictive(n_draws=5, random_state=91)
    default = result.sample_predictive(n_draws=5)

    pd.testing.assert_frame_equal(first, second)
    assert not first["outcome"].equals(default["outcome"])


def test_predictive_draws_condition_on_posterior_mean_parameters():
    result = _fit_family("gaussian")
    seed = 712
    n_draws = 3

    actual = result.sample_predictive(n_draws=n_draws, random_state=seed)
    parameters = {
        name: value.detach().cpu().numpy()
        for name, value in result.distributions.parameters.items()
    }
    rng = np.random.default_rng(seed)
    expected = []
    for i in range(result.panel.n):
        for j in range(result.panel.T):
            expected.extend(
                rng.normal(
                    loc=parameters["mean"][i, j],
                    scale=np.sqrt(parameters["variance"][i, j]),
                    size=n_draws,
                )
            )

    np.testing.assert_array_equal(actual["outcome"].to_numpy(), expected)


def test_predictive_sampling_accepts_maximum_unsigned_32_bit_seed():
    result = _fit_family("gaussian")

    first = result.sample_predictive(n_draws=2, random_state=2**32 - 1)
    second = result.sample_predictive(n_draws=2, random_state=2**32 - 1)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_draws": 0}, "n_draws"),
        ({"n_draws": True}, "n_draws"),
        ({"n_draws": 1.5}, "n_draws"),
        ({"n_draws": 1, "cells": "other"}, "cells"),
        ({"n_draws": 1, "random_state": -1}, "random_state"),
        ({"n_draws": 1, "random_state": True}, "random_state"),
        ({"n_draws": 1, "random_state": 1.5}, "random_state"),
        ({"n_draws": 1, "random_state": 2**32}, "random_state"),
    ],
)
def test_predictive_sampling_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _fit_family("gaussian").sample_predictive(**kwargs)


def _neighbor_panel_frame():
    rows = []
    for value in [1.0, 2.0]:
        rows.append({"state": "A", "year": 1, "y": value})
    for value in [3.0, 4.0, 5.0, 6.0]:
        rows.append({"state": "A", "year": 3, "y": value})
    for value in [7.0, 8.0, 9.0, 10.0, 11.0]:
        rows.append({"state": "B", "year": 2, "y": value})
    rows.append({"state": "B", "year": 3, "y": 12.0})
    return pd.DataFrame(rows)


def _fit_neighbor_panel():
    return EFSC(
        family="gaussian",
        rank=1,
        inference_config=InferenceConfig(epochs=5, learning_rate=0.03),
    ).fit(
        _neighbor_panel_frame(),
        outcome="y",
        unit="state",
        time="year",
    )


def test_complete_panel_with_fixed_cell_size_preserves_observations():
    source = _neighbor_panel_frame()
    result = _fit_neighbor_panel()
    completed = result.complete_panel(cell_size=2)

    assert completed.columns.tolist() == ["unit", "time", "outcome", "imputed"]
    observed = completed.loc[~completed["imputed"]].reset_index(drop=True)
    expected = source.rename(
        columns={"state": "unit", "year": "time", "y": "outcome"}
    )
    expected["imputed"] = False
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)

    imputed = completed.loc[completed["imputed"]]
    assert len(imputed) == 4
    assert imputed.groupby(["unit", "time"]).size().to_dict() == {
        ("A", 2): 2,
        ("B", 1): 2,
    }


def test_temporal_neighbor_cell_sizes_use_only_the_same_unit():
    result = _fit_neighbor_panel()
    completed = result.complete_panel(cell_size="temporal_neighbor_mean")
    imputed = completed.loc[completed["imputed"]]

    assert imputed.groupby(["unit", "time"]).size().to_dict() == {
        ("A", 2): 3,
        ("B", 1): 5,
    }
    pd.testing.assert_frame_equal(
        completed,
        result.complete_panel(cell_size="temporal_neighbor_mean"),
    )


@pytest.mark.parametrize(
    "cell_size",
    [0, -1, 1.5, True, "unit_neighbor", None],
)
def test_complete_panel_rejects_invalid_cell_size(cell_size):
    with pytest.raises(ValueError, match="cell_size"):
        _fit_neighbor_panel().complete_panel(cell_size=cell_size)
