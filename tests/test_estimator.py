import numpy as np
import pandas as pd
import pytest
import torch

from efsc import CausalResult, EFSC, EFSCResult, InferenceConfig


def _gaussian_frame():
    return pd.DataFrame(
        {
            "state": ["B", "A", "A", "A", "B", "B"],
            "year": [3, 1, 1, 2, 1, 3],
            "outcome": [4.0, 1.0, 2.0, 3.0, 2.5, 4.5],
        }
    )


def test_rank_is_required_and_random_state_defaults_to_zero():
    with pytest.raises(TypeError, match="rank"):
        EFSC(family="gaussian")

    model = EFSC(family="gaussian", rank=1)
    assert model.rank == 1
    assert model.random_state == 0
    assert model.inference_config == InferenceConfig()
    assert model.result_ is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 0}, "rank"),
        ({"rank": True}, "rank"),
        ({"rank": 1.5}, "rank"),
        ({"rank": 1, "random_state": -1}, "random_state"),
        ({"rank": 1, "random_state": True}, "random_state"),
        ({"rank": 1, "random_state": 1.5}, "random_state"),
        ({"rank": 1, "random_state": 2**32}, "random_state"),
        ({"rank": 1, "inference_config": {}}, "InferenceConfig"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        EFSC(family="gaussian", **kwargs)


@pytest.mark.parametrize("random_state", [0, np.int64(17), 2**32 - 1])
def test_constructor_accepts_full_unsigned_32_bit_seed_range(random_state):
    model = EFSC(family="gaussian", rank=1, random_state=random_state)

    assert model.random_state == int(random_state)


def test_derived_seeds_wrap_within_unsigned_32_bit_range():
    result = EFSC(
        family="gaussian",
        rank=1,
        random_state=2**32 - 1,
        inference_config=InferenceConfig(epochs=1),
    ).fit(
        _gaussian_frame(),
        outcome="outcome",
        unit="state",
        time="year",
    )

    assert result.factorization.seed == 2**32 - 1
    assert result.derived_seeds == {
        "factorization": 2**32 - 1,
        "predictive": 0,
    }


def test_dataframe_fit_returns_complete_imputation_result():
    model = EFSC(
        family="gaussian",
        rank=1,
        inference_config=InferenceConfig(epochs=20, learning_rate=0.03),
    )
    result = model.fit(
        _gaussian_frame(),
        outcome="outcome",
        unit="state",
        time="year",
    )

    assert isinstance(result, EFSCResult)
    assert model.result_ is result
    assert result.mode == "imputation"
    assert result.causal is None
    assert len(result.cells) == 6
    assert result.imputations[["unit", "time"]].to_records(index=False).tolist() == [
        ("A", 3),
        ("B", 2),
    ]
    assert result.category_probabilities is None
    assert result.factorization.seed == 0
    assert result.factorization.mask.tolist() == result.panel.observed_mask.tolist()
    assert len(result.factorization.loss_history) == 20
    assert result.derived_seeds == {"factorization": 0, "predictive": 1}
    assert result.configuration["family"] == "gaussian"
    assert result.configuration["mode"] == "imputation"
    assert result.configuration["rank"] == 1
    assert result.configuration["random_state"] == 0
    assert result.configuration["inference"]["epochs"] == 20
    assert set(result.software_versions) == {
        "python",
        "numpy",
        "pandas",
        "torch",
    }


def test_csv_and_dataframe_paths_are_fixed_seed_equivalent(tmp_path):
    path = tmp_path / "panel.csv"
    _gaussian_frame().to_csv(path, index=False)
    config = InferenceConfig(epochs=15, learning_rate=0.03)
    frame_result = EFSC(
        family="gaussian",
        rank=1,
        random_state=73,
        inference_config=config,
    ).fit(
        _gaussian_frame(),
        outcome="outcome",
        unit="state",
        time="year",
    )
    csv_result = EFSC(
        family="gaussian",
        rank=1,
        random_state=73,
        inference_config=config,
    ).fit_csv(
        path,
        outcome="outcome",
        unit="state",
        time="year",
    )

    torch.testing.assert_close(
        csv_result.factorization.variational_parameters,
        frame_result.factorization.variational_parameters,
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        csv_result.factorization.loss_history,
        frame_result.factorization.loss_history,
    )
    pd.testing.assert_frame_equal(csv_result.cells, frame_result.cells)


def test_multinomial_fit_exposes_ordered_category_probabilities():
    frame = pd.DataFrame(
        {
            "unit": ["A", "A", "A", "B", "B", "B"],
            "time": [1, 1, 2, 1, 2, 2],
            "insurance": ["x", "y", "z", "z", "x", "x"],
        }
    )
    result = EFSC(
        family="multinomial",
        rank=1,
        categories=["x", "y", "z"],
        inference_config=InferenceConfig(epochs=15),
    ).fit(
        frame,
        outcome="insurance",
        unit="unit",
        time="time",
    )

    assert result.imputations.empty
    probabilities = result.category_probabilities
    assert probabilities is not None
    assert len(probabilities) == 12
    assert probabilities["category"].drop_duplicates().tolist() == ["x", "y", "z"]
    assert probabilities.loc[
        probabilities["reference_category"], "category"
    ].unique().tolist() == ["z"]
    np.testing.assert_allclose(
        probabilities.groupby(["unit_index", "time_index"])["probability"].sum(),
        1.0,
    )


def _causal_gaussian_frame():
    return pd.DataFrame(
        {
            "state": ["A", "A", "B", "B", "B"],
            "year": [1, 2, 1, 2, 3],
            "outcome": [0.5, 1.5, -0.5, 0.0, 0.75],
        }
    )


def test_causal_fit_uses_full_coordinates_and_separate_masks():
    model = EFSC(
        family="gaussian",
        rank=1,
        random_state=17,
        inference_config=InferenceConfig(epochs=10, learning_rate=0.03),
    )
    result = model.fit(
        _causal_gaussian_frame(),
        outcome="outcome",
        unit="state",
        time="year",
        causal_metadata={"A": 2, "B": "never"},
    )

    assert isinstance(result, EFSCResult)
    assert isinstance(result.causal, CausalResult)
    assert result.mode == "causal"
    assert model.result_ is result
    assert result.factorization is result.causal.counterfactual_fit
    assert result.distributions is result.causal.counterfactual_distributions
    assert result.factorization.natural_parameters.shape[:2] == (2, 3)
    assert result.causal.treated_fit.natural_parameters.shape[:2] == (2, 3)
    np.testing.assert_array_equal(
        result.factorization.mask,
        result.causal.masks.counterfactual_mask,
    )
    np.testing.assert_array_equal(
        result.causal.treated_fit.mask,
        result.causal.masks.treated_mask,
    )
    assert result.cells["role"].tolist() == [
        "counterfactual_training",
        "treated_post_observed",
        "treated_post_missing",
        "counterfactual_training",
        "counterfactual_training",
        "counterfactual_training",
    ]
    assert result.causal.counterfactuals[["unit", "time"]].to_records(
        index=False
    ).tolist() == [("A", 2), ("A", 3)]
    assert result.causal.effects[["unit", "time"]].to_records(
        index=False
    ).tolist() == [("A", 2)]
    assert result.derived_seeds == {
        "counterfactual": 17,
        "treated": 18,
        "posterior_effects": 19,
        "predictive": 20,
    }
    assert result.configuration["mode"] == "causal"
    assert result.configuration["causal"] == {
        "adoption_times": {"A": 2, "B": None},
        "treated_fit_scope": "full",
        "counterfactual_training_cells": 4,
        "treated_training_cells": 1,
        "treated_post_targets": 2,
        "missing_treated_post_targets": 1,
    }


def test_causal_csv_and_dataframe_paths_are_fixed_seed_equivalent(tmp_path):
    panel_path = tmp_path / "panel.csv"
    metadata_path = tmp_path / "treatment.csv"
    _causal_gaussian_frame().to_csv(panel_path, index=False)
    pd.DataFrame(
        {
            "unit": ["A", "B"],
            "adoption_time": [2, "never"],
        }
    ).to_csv(metadata_path, index=False)
    config = InferenceConfig(epochs=8, learning_rate=0.03)

    frame_result = EFSC(
        family="gaussian",
        rank=1,
        random_state=29,
        inference_config=config,
    ).fit(
        _causal_gaussian_frame(),
        outcome="outcome",
        unit="state",
        time="year",
        causal_metadata={"A": 2, "B": "never"},
    )
    csv_result = EFSC(
        family="gaussian",
        rank=1,
        random_state=29,
        inference_config=config,
    ).fit_csv(
        panel_path,
        outcome="outcome",
        unit="state",
        time="year",
        causal_metadata=metadata_path,
    )

    torch.testing.assert_close(
        csv_result.causal.counterfactual_fit.variational_parameters,
        frame_result.causal.counterfactual_fit.variational_parameters,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        csv_result.causal.treated_fit.variational_parameters,
        frame_result.causal.treated_fit.variational_parameters,
        rtol=0,
        atol=0,
    )
    pd.testing.assert_frame_equal(csv_result.cells, frame_result.cells)
    pd.testing.assert_frame_equal(
        csv_result.causal.effects,
        frame_result.causal.effects,
    )


def test_causal_predictive_sampling_uses_counterfactual_distribution():
    result = EFSC(
        family="gaussian",
        rank=1,
        random_state=41,
        inference_config=InferenceConfig(epochs=5, learning_rate=0.03),
    ).fit(
        _causal_gaussian_frame(),
        outcome="outcome",
        unit="state",
        time="year",
        causal_metadata={"A": 2, "B": "never"},
    )

    assert result.distributions is result.causal.counterfactual_distributions
    first = result.sample_predictive(n_draws=3)
    second = result.sample_predictive(n_draws=3)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 18


@pytest.mark.parametrize(
    ("family", "values", "kwargs"),
    [
        ("gaussian", [0.2, 0.4, 0.8, -0.1, 0.1, 0.3], {}),
        (
            "gaussian_known_sigma",
            [0.2, 0.4, 0.8, -0.1, 0.1, 0.3],
            {"sigma": 1.5},
        ),
        ("poisson", [0, 1, 2, 1, 2, 3], {}),
        ("bernoulli", [0, 1, 1, 1, 0, 1], {}),
        ("exponential", [0.2, 0.4, 0.8, 0.3, 0.6, 1.0], {}),
        (
            "laplace_known_mu",
            [0.2, 0.4, 0.8, -0.1, 0.1, 0.3],
            {"mu": 0.0},
        ),
        ("chi_square", [0.2, 0.4, 0.8, 0.3, 0.6, 1.0], {}),
        (
            "multinomial",
            ["a", "b", "c", "c", "a", "b"],
            {"categories": ["a", "b", "c"]},
        ),
    ],
)
def test_causal_estimator_smoke_supports_every_family(family, values, kwargs):
    frame = pd.DataFrame(
        {
            "unit": ["A", "A", "A", "B", "B", "B"],
            "time": [1, 2, 3, 1, 2, 3],
            "outcome": values,
        }
    )
    result = EFSC(
        family=family,
        rank=1,
        inference_config=InferenceConfig(epochs=2),
        **kwargs,
    ).fit(
        frame,
        outcome="outcome",
        unit="unit",
        time="time",
        causal_metadata={"A": 3, "B": "never"},
    )

    assert result.mode == "causal"
    assert len(result.causal.counterfactuals) == 1
    assert len(result.causal.effects) == 1
    assert np.isfinite(result.causal.effects["ecd"]).all()
    assert (result.causal.effects["ecd"] >= -1e-7).all()
    if family == "multinomial":
        assert len(result.causal.category_effects) == 3
    else:
        assert result.causal.category_effects is None
