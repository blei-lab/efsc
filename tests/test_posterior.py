import numpy as np
import pandas as pd
import pytest
import torch

from efsc import EFSC, InferenceConfig


def _causal_result(family="bernoulli", *, values=None, **kwargs):
    if values is None:
        values = [0, 1, 1, 1, 0, 1]
    frame = pd.DataFrame(
        {
            "unit": ["A", "A", "A", "B", "B", "B"],
            "time": [1, 2, 3, 1, 2, 3],
            "outcome": values,
        }
    )
    return EFSC(
        family=family,
        rank=1,
        random_state=13,
        inference_config=InferenceConfig(epochs=2),
        **kwargs,
    ).fit(
        frame,
        outcome="outcome",
        unit="unit",
        time="time",
        causal_metadata={"A": 3, "B": "never"},
    )


@pytest.fixture(scope="module")
def bernoulli_result():
    return _causal_result()


def test_posterior_effect_draws_are_deterministic_and_cell_level(bernoulli_result):
    causal = bernoulli_result.causal
    first = causal.sample_effects(n_draws=20)
    second = causal.sample_effects(n_draws=20)
    explicit = causal.sample_effects(
        n_draws=20,
        random_state=causal.posterior_effects_seed,
    )
    different = causal.sample_effects(n_draws=20, random_state=99)

    assert first.columns.tolist() == [
        "unit",
        "time",
        "unit_index",
        "time_index",
        "observation_count",
        "draw",
        "ece_eta_0",
        "ecd",
    ]
    assert len(first) == 20
    assert first[["unit", "time"]].drop_duplicates().to_records(
        index=False
    ).tolist() == [("A", 3)]
    assert first["draw"].tolist() == list(range(1, 21))
    assert np.isfinite(first[["ece_eta_0", "ecd"]]).all().all()
    assert (first["ecd"] >= -1e-7).all()
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, explicit)
    assert not first[["ece_eta_0", "ecd"]].equals(
        different[["ece_eta_0", "ecd"]]
    )


def test_posterior_sampling_accepts_maximum_unsigned_32_bit_seed(
    bernoulli_result,
):
    causal = bernoulli_result.causal

    first = causal.sample_effects(n_draws=2, random_state=2**32 - 1)
    second = causal.sample_effects(n_draws=2, random_state=2**32 - 1)

    pd.testing.assert_frame_equal(first, second)


def test_posterior_effect_summary_matches_the_same_draws(bernoulli_result):
    causal = bernoulli_result.causal
    draws = causal.sample_effects(n_draws=40)
    summary = causal.summarize_effects(n_draws=40, level=0.8)
    lower_probability = 0.1
    upper_probability = 0.9

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_draws"] == 40
    assert row["level"] == 0.8
    for column in ["ece_eta_0", "ecd"]:
        values = draws[column].to_numpy()
        np.testing.assert_allclose(row[f"{column}_mean"], values.mean())
        np.testing.assert_allclose(row[f"{column}_sd"], values.std(ddof=1))
        np.testing.assert_allclose(
            row[f"{column}_lower"],
            np.quantile(values, lower_probability),
        )
        np.testing.assert_allclose(
            row[f"{column}_upper"],
            np.quantile(values, upper_probability),
        )


@pytest.mark.parametrize(
    ("family", "values", "kwargs", "parameter_count"),
    [
        ("gaussian", [0.2, 0.4, 0.8, -0.1, 0.1, 0.3], {}, 2),
        (
            "gaussian_known_sigma",
            [0.2, 0.4, 0.8, -0.1, 0.1, 0.3],
            {"sigma": 1.5},
            1,
        ),
        ("poisson", [0, 1, 2, 1, 2, 3], {}, 1),
        ("bernoulli", [0, 1, 1, 1, 0, 1], {}, 1),
        ("exponential", [0.2, 0.4, 0.8, 0.3, 0.6, 1.0], {}, 1),
        (
            "laplace_known_mu",
            [0.2, 0.4, 0.8, -0.1, 0.1, 0.3],
            {"mu": 0.0},
            1,
        ),
        ("chi_square", [0.2, 0.4, 0.8, 0.3, 0.6, 1.0], {}, 1),
        (
            "multinomial",
            ["a", "b", "c", "c", "a", "b"],
            {"categories": ["a", "b", "c"]},
            2,
        ),
    ],
)
def test_posterior_effect_sampling_supports_every_family(
    family,
    values,
    kwargs,
    parameter_count,
):
    result = _causal_result(family, values=values, **kwargs)

    draws = result.causal.sample_effects(n_draws=3)

    assert len(draws) == 3
    assert [column for column in draws if column.startswith("ece_eta_")] == [
        f"ece_eta_{index}" for index in range(parameter_count)
    ]
    assert np.isfinite(draws.filter(regex=r"^(ece_eta_|ecd$)")).all().all()


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        ("sample", {"n_draws": 0}, "n_draws"),
        ("sample", {"n_draws": True}, "n_draws"),
        ("sample", {"n_draws": 1.5}, "n_draws"),
        ("sample", {"n_draws": 2, "random_state": -1}, "random_state"),
        ("sample", {"n_draws": 2, "random_state": True}, "random_state"),
        ("sample", {"n_draws": 2, "random_state": 1.5}, "random_state"),
        ("sample", {"n_draws": 2, "random_state": 2**32}, "random_state"),
        ("summary", {"n_draws": 1}, "at least two"),
        ("summary", {"n_draws": 1.5}, "n_draws"),
        ("summary", {"n_draws": 2, "level": 0}, "level"),
        ("summary", {"n_draws": 2, "level": 1}, "level"),
    ],
)
def test_posterior_effect_sampling_rejects_invalid_options(
    bernoulli_result,
    method,
    kwargs,
    message,
):
    function = (
        bernoulli_result.causal.sample_effects
        if method == "sample"
        else bernoulli_result.causal.summarize_effects
    )
    with pytest.raises(ValueError, match=message):
        function(**kwargs)


def test_posterior_effect_sampling_does_not_change_global_torch_rng(
    bernoulli_result,
):
    torch.manual_seed(812)
    state_before = torch.random.get_rng_state().clone()

    bernoulli_result.causal.sample_effects(n_draws=5)

    torch.testing.assert_close(torch.random.get_rng_state(), state_before)
