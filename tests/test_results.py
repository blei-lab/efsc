import numpy as np
import pandas as pd
import pytest
import torch

from efsc.fitting import FactorizationFit, InferenceConfig
from efsc.panel import prepare_panel_data
from efsc.results import (
    build_factorization_tables,
    distribution_parameters_from_eta,
)

@pytest.mark.parametrize(
    ("family", "eta", "kwargs", "message"),
    [
        ("gaussian", torch.zeros((2, 2, 1)), {}, "two"),
        (
            "gaussian",
            torch.zeros((2, 2, 2)),
            {},
            "second natural parameter",
        ),
        ("exponential", torch.ones((2, 2, 1)), {}, "negative"),
        ("laplace_known_mu", -torch.ones((2, 2, 1)), {}, "mu"),
        ("chi_square", -torch.ones((2, 2, 1)), {}, "greater than -1"),
        (
            "multinomial",
            torch.zeros((2, 2, 2)),
            {"categories": ["a", "b"]},
            "categories",
        ),
        (
            "multinomial",
            torch.zeros((2, 2, 2)),
            {"categories": ["a", "a", "reference"]},
            "unique",
        ),
    ],
)
def test_distribution_conversion_rejects_invalid_inputs(
    family,
    eta,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        distribution_parameters_from_eta(eta, family=family, **kwargs)


def _gaussian_panel():
    return prepare_panel_data(
        pd.DataFrame(
            {
                "state": ["B", "A", "A", "B"],
                "year": [3, 1, 2, 1],
                "outcome": [4.0, 1.0, 2.0, 3.0],
            }
        ),
        outcome="outcome",
        unit="state",
        time="year",
        family="gaussian",
    )


def _fake_fit(panel, eta):
    dimension = 4
    return FactorizationFit(
        family=panel.family,
        rank=1,
        seed=0,
        mask=panel.observed_mask.copy(),
        variational_mean=torch.zeros(dimension),
        variational_log_variance=torch.zeros(dimension),
        raw_natural_parameters=eta.clone(),
        natural_parameters=eta,
        loss_history=np.array([1.0]),
        elapsed_time=0.0,
        learning_rate=0.1,
        config=InferenceConfig(epochs=1),
    )


def test_cell_and_imputation_tables_preserve_panel_structure():
    panel = _gaussian_panel()
    eta0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    eta1 = -torch.ones((2, 3), dtype=torch.float32)
    eta = torch.stack([eta0, eta1], dim=2)

    tables = build_factorization_tables(panel, _fake_fit(panel, eta))

    assert tables.cells.columns.tolist() == [
        "unit",
        "time",
        "unit_index",
        "time_index",
        "observation_count",
        "observed",
        "role",
        "eta_0",
        "eta_1",
        "mean",
        "variance",
    ]
    assert tables.cells[["unit", "time"]].to_records(index=False).tolist() == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("B", 1),
        ("B", 2),
        ("B", 3),
    ]
    np.testing.assert_array_equal(
        tables.cells["observation_count"],
        [1, 1, 0, 1, 0, 1],
    )
    assert tables.cells["role"].tolist() == [
        "observed",
        "observed",
        "missing",
        "observed",
        "missing",
        "observed",
    ]
    assert tables.imputations[["unit", "time"]].to_records(index=False).tolist() == [
        ("A", 3),
        ("B", 2),
    ]
    np.testing.assert_allclose(tables.cells["mean"], eta0.numpy().reshape(-1) / 2)
    np.testing.assert_allclose(tables.cells["variance"], 0.5)
    assert tables.category_probabilities is None


def test_custom_roles_are_preserved_in_cell_tables():
    panel = _gaussian_panel()
    eta = torch.stack(
        [torch.zeros((2, 3)), -torch.ones((2, 3))],
        dim=2,
    )
    roles = np.array(
        [
            ["counterfactual_training", "counterfactual_training", "target"],
            ["counterfactual_training", "missing", "treated_post_observed"],
        ],
        dtype=object,
    )

    tables = build_factorization_tables(
        panel,
        _fake_fit(panel, eta),
        roles=roles,
    )
    assert tables.cells["role"].tolist() == roles.reshape(-1).tolist()


def test_cell_tables_reject_missing_custom_roles():
    panel = _gaussian_panel()
    eta = torch.stack(
        [torch.zeros((2, 3)), -torch.ones((2, 3))],
        dim=2,
    )
    roles = np.full((2, 3), "observed", dtype=object)
    roles[0, 2] = np.nan

    with pytest.raises(ValueError, match="roles"):
        build_factorization_tables(panel, _fake_fit(panel, eta), roles=roles)


def test_multinomial_tables_include_wide_and_long_probabilities():
    panel = prepare_panel_data(
        pd.DataFrame(
            {
                "unit": ["A", "A", "B", "B"],
                "time": [1, 2, 1, 2],
                "y": ["x", "y", "z", "x"],
            }
        ),
        outcome="y",
        unit="unit",
        time="time",
        family="multinomial",
        categories=["x", "y", "z"],
    )
    eta = torch.tensor(
        [
            [[0.2, -0.1], [0.4, 0.3]],
            [[-0.2, 0.1], [0.5, -0.4]],
        ]
    )
    tables = build_factorization_tables(panel, _fake_fit(panel, eta))

    wide = tables.cells[["probability_0", "probability_1", "probability_2"]]
    np.testing.assert_allclose(wide.sum(axis=1), 1.0)
    long = tables.category_probabilities
    assert long is not None
    assert long.columns.tolist() == [
        "unit",
        "time",
        "unit_index",
        "time_index",
        "category",
        "category_index",
        "reference_category",
        "probability",
        "observed",
        "role",
    ]
    assert len(long) == 12
    assert long.loc[long["reference_category"], "category"].unique().tolist() == [
        "z"
    ]
    np.testing.assert_allclose(
        long.groupby(["unit_index", "time_index"])["probability"].sum(),
        1.0,
        rtol=1e-6,
        atol=1e-7,
    )


def test_table_builder_rejects_fit_from_different_family():
    panel = _gaussian_panel()
    eta = torch.stack(
        [torch.zeros((2, 3)), -torch.ones((2, 3))],
        dim=2,
    )
    fit = _fake_fit(panel, eta)
    fit = FactorizationFit(
        **{**fit.__dict__, "family": "poisson"},
    )

    with pytest.raises(ValueError, match="family"):
        build_factorization_tables(panel, fit)
