from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from efsc.panel import (
    build_causal_masks,
    prepare_panel_data,
    read_panel_data_csv,
)


def _numeric_panel():
    return pd.DataFrame(
        {
            "unit": ["B", "A", "A", "A", "B"],
            "time": [2012, 2010, 2010, 2011, 2010],
            "y": [5.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


def _complete_panel(units=("A", "B", "C"), times=(1, 2, 3, 4)):
    return pd.DataFrame(
        [
            {"unit": unit, "time": time, "y": float(i + j)}
            for i, unit in enumerate(units)
            for j, time in enumerate(times)
        ]
    )


def test_prepare_panel_preserves_repetitions_and_constructs_missing_cells():
    panel = prepare_panel_data(
        _numeric_panel(),
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )

    assert panel.family == "gaussian"
    assert panel.unit_labels == ("A", "B")
    assert panel.time_labels == (2010, 2011, 2012)
    assert panel.time_kind == "numeric"
    assert panel.time_step == 1
    np.testing.assert_array_equal(
        panel.observed_mask,
        [[True, True, False], [True, False, True]],
    )
    np.testing.assert_array_equal(panel.cell_counts, [[2, 1, 0], [1, 0, 1]])
    np.testing.assert_allclose(panel.observations[0, 0], [1.0, 2.0])
    assert panel.observations[0, 2].size == 0


def test_read_panel_csv_matches_dataframe_entry_point(tmp_path):
    path = tmp_path / "panel.csv"
    _numeric_panel().to_csv(path, index=False)

    from_csv = read_panel_data_csv(
        path,
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )
    from_frame = prepare_panel_data(
        _numeric_panel(),
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )

    assert from_csv.unit_labels == from_frame.unit_labels
    assert from_csv.time_labels == from_frame.time_labels
    np.testing.assert_array_equal(from_csv.cell_counts, from_frame.cell_counts)


def test_numeric_time_requires_constant_spacing():
    frame = pd.DataFrame(
        {"unit": ["A", "A", "A"], "time": [2010, 2011, 2013], "y": [1, 2, 3]}
    )

    with pytest.raises(ValueError, match="equally spaced"):
        prepare_panel_data(
            frame,
            outcome="y",
            unit="unit",
            time="time",
            family="gaussian",
        )


def test_arbitrary_time_labels_require_complete_explicit_order():
    frame = pd.DataFrame(
        {"unit": ["A", "A", "A"], "time": ["late", "early", "middle"], "y": [3, 1, 2]}
    )

    with pytest.raises(ValueError, match="time_order"):
        prepare_panel_data(
            frame,
            outcome="y",
            unit="unit",
            time="time",
            family="gaussian",
        )

    panel = prepare_panel_data(
        frame,
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
        time_order=["early", "middle", "late"],
    )
    assert panel.time_labels == ("early", "middle", "late")
    assert panel.time_kind == "ordered"
    assert panel.time_step is None

    with pytest.raises(ValueError, match="exactly once"):
        prepare_panel_data(
            frame,
            outcome="y",
            unit="unit",
            time="time",
            family="gaussian",
            time_order=["early", "late"],
        )


def test_parseable_dates_are_sorted_chronologically():
    frame = pd.DataFrame(
        {
            "unit": ["A", "A", "A"],
            "time": ["2020-03-01", "2020-01-01", "2020-02-01"],
            "y": [3, 1, 2],
        }
    )
    panel = prepare_panel_data(
        frame,
        outcome="y",
        unit="unit",
        time="time",
        family="poisson",
    )

    assert panel.time_labels == ("2020-01-01", "2020-02-01", "2020-03-01")
    assert panel.time_kind == "datetime"
    assert panel.time_step is None


@pytest.mark.parametrize(
    ("family", "values", "kwargs", "message"),
    [
        ("poisson", [0, 1.5], {}, "nonnegative integers"),
        ("bernoulli", [0, 2], {}, "0 or 1"),
        ("exponential", [0, -1], {}, "nonnegative"),
        ("chi_square", [1, 0], {}, "strictly positive"),
        ("gaussian_known_sigma", [1, 2], {}, "sigma"),
        ("gaussian_known_sigma", [1, 2], {"sigma": 0}, "sigma"),
        ("laplace_known_mu", [1, 2], {}, "mu"),
    ],
)
def test_family_specific_validation(family, values, kwargs, message):
    frame = pd.DataFrame({"unit": ["A", "A"], "time": [1, 2], "y": values})

    with pytest.raises(ValueError, match=message):
        prepare_panel_data(
            frame,
            outcome="y",
            unit="unit",
            time="time",
            family=family,
            **kwargs,
        )


@pytest.mark.parametrize("column", ["unit", "time", "y"])
def test_blank_or_missing_required_values_raise(column):
    frame = pd.DataFrame(
        {"unit": ["A", "B"], "time": [1, 2], "y": [1.0, 2.0]}
    )
    if column != "y":
        frame[column] = frame[column].astype(object)
    frame.loc[1, column] = np.nan if column == "y" else " "

    with pytest.raises(ValueError, match=column):
        prepare_panel_data(
            frame,
            outcome="y",
            unit="unit",
            time="time",
            family="gaussian",
        )


def test_multinomial_requires_and_preserves_explicit_category_order():
    frame = pd.DataFrame(
        {
            "unit": ["A", "A", "A", "B"],
            "time": [1, 1, 2, 2],
            "y": ["medicaid", "other", "private", "medicaid"],
        }
    )

    with pytest.raises(ValueError, match="categories"):
        prepare_panel_data(
            frame,
            outcome="y",
            unit="unit",
            time="time",
            family="multinomial",
        )

    panel = prepare_panel_data(
        frame,
        outcome="y",
        unit="unit",
        time="time",
        family="multinomial",
        categories=["medicaid", "private", "other"],
    )
    assert panel.categories == ("medicaid", "private", "other")
    assert panel.reference_category == "other"
    assert panel.observations[0, 0].tolist() == ["medicaid", "other"]

    with pytest.raises(ValueError, match="not listed"):
        prepare_panel_data(
            frame,
            outcome="y",
            unit="unit",
            time="time",
            family="multinomial",
            categories=["medicaid", "other"],
        )


def test_legacy_family_names_are_normalized():
    frame = pd.DataFrame({"unit": ["A"], "time": [1], "y": [2.0]})
    panel = prepare_panel_data(
        frame,
        outcome="y",
        unit="unit",
        time="time",
        family="gauss_univariate",
    )
    assert panel.family == "gaussian"


def test_absorbing_treatment_masks_respect_missing_treated_post_cells():
    frame = _complete_panel()
    frame = frame[~((frame["unit"] == "A") & (frame["time"] == 4))]
    panel = prepare_panel_data(
        frame,
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )

    masks = build_causal_masks(panel, {"A": 3, "B": 4})

    np.testing.assert_array_equal(
        masks.treated_post_mask,
        [
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ],
    )
    np.testing.assert_array_equal(
        masks.counterfactual_mask,
        panel.observed_mask & ~masks.treated_post_mask,
    )
    np.testing.assert_array_equal(
        masks.treated_mask,
        panel.observed_mask & masks.treated_post_mask,
    )
    assert masks.treated_post_mask[0, 3]
    assert not masks.treated_mask[0, 3]
    assert masks.adoption_times == {"A": 3, "B": 4, "C": None}


@pytest.mark.parametrize("never_value", [None, np.nan, "", "  ", "never", "NEVER"])
def test_blank_and_never_metadata_values_mean_never_treated(never_value):
    panel = prepare_panel_data(
        _complete_panel(units=("A", "B")),
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )
    metadata = pd.DataFrame(
        {"unit": ["A", "B"], "adoption_time": [3, never_value]}
    )

    masks = build_causal_masks(panel, metadata)
    assert masks.adoption_times == {"A": 3, "B": None}
    assert not masks.treated_post_mask[1].any()


def test_causal_metadata_can_be_read_from_csv(tmp_path):
    panel = prepare_panel_data(
        _complete_panel(units=("A", "B")),
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )
    path = tmp_path / "treatment.csv"
    pd.DataFrame(
        {"unit": ["A", "B"], "adoption_time": [3, "never"]}
    ).to_csv(path, index=False)

    masks = build_causal_masks(panel, Path(path))
    assert masks.adoption_times == {"A": 3, "B": None}


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"unknown": 3}, "not present"),
        ({"A": 99}, "panel time"),
        ({"A": 1}, "pretreatment"),
        ({"A": 4}, "treated-post"),
    ],
)
def test_invalid_causal_metadata_raise(metadata, message):
    panel = prepare_panel_data(
        _complete_panel(),
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )
    if metadata == {"A": 4}:
        panel.observed_mask[0, 3] = False

    with pytest.raises(ValueError, match=message):
        build_causal_masks(panel, metadata)


def test_duplicate_causal_metadata_units_raise():
    panel = prepare_panel_data(
        _complete_panel(),
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )
    metadata = pd.DataFrame(
        {"unit": ["A", "A"], "adoption_time": [3, 4]}
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_causal_masks(panel, metadata)


def test_counterfactual_fit_must_retain_every_time_period():
    panel = prepare_panel_data(
        _complete_panel(units=("A", "B"), times=(1, 2, 3)),
        outcome="y",
        unit="unit",
        time="time",
        family="gaussian",
    )

    with pytest.raises(ValueError, match="counterfactual.*time"):
        build_causal_masks(panel, {"A": 3, "B": 3})
