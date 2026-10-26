import numpy as np
import pandas as pd
import pytest
import torch

from efsc.causal_results import build_causal_result
from efsc.fitting import FactorizationFit, InferenceConfig
from efsc.panel import build_causal_masks, prepare_panel_data


def _panel_frame(outcomes):
    rows = []
    for unit, values in outcomes.items():
        for time, outcome in enumerate(values, start=1):
            if outcome is not None:
                rows.append({"state": unit, "year": time, "outcome": outcome})
    return pd.DataFrame(rows)


def _fake_fit(panel, eta, mask, seed=0, rank=1):
    dimension = eta.numel()
    return FactorizationFit(
        family=panel.family,
        rank=rank,
        seed=seed,
        mask=mask.copy(),
        variational_mean=torch.zeros(dimension),
        variational_log_variance=torch.zeros(dimension),
        raw_natural_parameters=eta.clone(),
        natural_parameters=eta,
        loss_history=np.array([1.0]),
        elapsed_time=0.0,
        learning_rate=0.1,
        config=InferenceConfig(epochs=1),
    )


def _bernoulli_inputs():
    panel = prepare_panel_data(
        _panel_frame(
            {
                "A": [0, 1, 1, None],
                "B": [1, 0, 1, 0],
            }
        ),
        outcome="outcome",
        unit="state",
        time="year",
        family="bernoulli",
    )
    masks = build_causal_masks(panel, {"A": 3, "B": "never"})
    counterfactual_eta = torch.tensor(
        [
            [[-0.8], [-0.4], [0.1], [0.3]],
            [[0.2], [0.4], [0.6], [0.8]],
        ],
        dtype=torch.float64,
    )
    treated_eta = counterfactual_eta + 0.7
    return panel, masks, counterfactual_eta, treated_eta


def test_causal_tables_separate_targets_from_available_effects():
    panel, masks, counterfactual_eta, treated_eta = _bernoulli_inputs()
    counterfactual_fit = _fake_fit(
        panel,
        counterfactual_eta,
        masks.counterfactual_mask,
        seed=5,
    )
    treated_fit = _fake_fit(panel, treated_eta, masks.treated_mask, seed=6)

    result = build_causal_result(
        panel,
        masks,
        counterfactual_fit,
        treated_fit,
    )

    assert result.counterfactuals[["unit", "time"]].to_records(
        index=False
    ).tolist() == [("A", 3), ("A", 4)]
    assert result.counterfactuals["effect_available"].tolist() == [True, False]
    assert result.counterfactuals["role"].tolist() == [
        "treated_post_observed",
        "treated_post_missing",
    ]
    assert result.effects[["unit", "time"]].to_records(index=False).tolist() == [
        ("A", 3)
    ]
    np.testing.assert_allclose(result.effects["ece_eta_0"], 0.7)
    np.testing.assert_allclose(
        result.effects["difference_probability"],
        result.effects["treated_probability"]
        - result.effects["counterfactual_probability"],
    )
    assert result.category_effects is None
    assert result.counterfactual_fit is counterfactual_fit
    assert result.treated_fit is treated_fit
    assert result.masks is masks


def test_gaussian_effects_include_both_formal_and_familiar_contrasts():
    panel = prepare_panel_data(
        _panel_frame(
            {
                "A": [0.2, 0.5, 1.0, 0.7],
                "B": [-0.1, 0.1, 0.3, 0.4],
            }
        ),
        outcome="outcome",
        unit="state",
        time="year",
        family="gaussian",
    )
    masks = build_causal_masks(panel, {"A": 3, "B": "never"})
    counterfactual_eta = torch.zeros((2, 4, 2), dtype=torch.float64)
    counterfactual_eta[..., 1] = -0.5
    treated_eta = counterfactual_eta.clone()
    treated_eta[0, 2:, 0] = 1.0
    treated_eta[0, 2:, 1] = -0.25

    result = build_causal_result(
        panel,
        masks,
        _fake_fit(panel, counterfactual_eta, masks.counterfactual_mask),
        _fake_fit(panel, treated_eta, masks.treated_mask, seed=1),
    )

    np.testing.assert_allclose(result.effects["ece_eta_0"], 1.0)
    np.testing.assert_allclose(result.effects["ece_eta_1"], 0.25)
    np.testing.assert_allclose(result.effects["counterfactual_mean"], 0.0)
    np.testing.assert_allclose(result.effects["treated_mean"], 2.0)
    np.testing.assert_allclose(result.effects["difference_mean"], 2.0)
    np.testing.assert_allclose(result.effects["counterfactual_variance"], 1.0)
    np.testing.assert_allclose(result.effects["treated_variance"], 2.0)
    np.testing.assert_allclose(result.effects["difference_variance"], 1.0)
    assert np.all(result.effects["ecd"] > 0)


def test_multinomial_category_effects_include_reference_category():
    panel = prepare_panel_data(
        _panel_frame(
            {
                "A": ["no", "yes", "yes", None],
                "B": ["other", "no", "yes", "other"],
            }
        ),
        outcome="outcome",
        unit="state",
        time="year",
        family="multinomial",
        categories=["no", "yes", "other"],
    )
    masks = build_causal_masks(panel, {"A": 3, "B": "never"})
    counterfactual_eta = torch.zeros((2, 4, 2), dtype=torch.float64)
    treated_eta = counterfactual_eta.clone()
    treated_eta[0, 2:, :] = torch.tensor([1.0, -0.5])

    result = build_causal_result(
        panel,
        masks,
        _fake_fit(panel, counterfactual_eta, masks.counterfactual_mask),
        _fake_fit(panel, treated_eta, masks.treated_mask, seed=1),
    )

    assert result.effects.columns.tolist()[-7:] == [
        "counterfactual_eta_0",
        "treated_eta_0",
        "ece_eta_0",
        "counterfactual_eta_1",
        "treated_eta_1",
        "ece_eta_1",
        "ecd",
    ]
    assert result.category_effects["category"].tolist() == ["no", "yes", "other"]
    assert result.category_effects["is_reference_category"].tolist() == [
        False,
        False,
        True,
    ]
    assert result.category_effects["reference_category"].tolist() == [
        "other",
        "other",
        "other",
    ]
    np.testing.assert_allclose(
        result.category_effects["difference_probability"].sum(),
        0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.counterfactuals[
            [
                "counterfactual_probability_0",
                "counterfactual_probability_1",
                "counterfactual_probability_2",
            ]
        ].sum(axis=1),
        1.0,
    )


def test_causal_result_requires_the_exact_approved_masks():
    panel, masks, counterfactual_eta, treated_eta = _bernoulli_inputs()
    wrong_treated_mask = panel.observed_mask.copy()

    with pytest.raises(ValueError, match="treated mask"):
        build_causal_result(
            panel,
            masks,
            _fake_fit(panel, counterfactual_eta, masks.counterfactual_mask),
            _fake_fit(panel, treated_eta, wrong_treated_mask, seed=1),
        )
