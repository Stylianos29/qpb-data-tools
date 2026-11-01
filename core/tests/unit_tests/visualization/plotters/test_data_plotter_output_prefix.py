"""
Unit tests for the ``output_prefix`` option of ``DataPlotter.plot()``.

``output_prefix`` is prepended to the saved plot's filename and to its
top-level folder, so that two plots of the same variables and grouping
(e.g. the same quantities for different sets of methods) do not
overwrite each other. It takes precedence over
``include_combined_prefix`` and can be set per figure through
``per_figure_overrides``.
"""

import os

import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing

from library.visualization.plotters.data_plotter import DataPlotter


@pytest.fixture
def plotter(tmp_path):
    """
    DataPlotter with two figures (one per Configuration_label), each
    with two curves grouped by Overlap_operator_method.
    """
    rows = []
    for config in ["0007200", "0018000"]:
        for method in ["Chebyshev", "Diagonal KL"]:
            for n in [2, 4, 6]:
                rows.append(
                    {
                        "Configuration_label": config,
                        "Overlap_operator_method": method,
                        "Number_of_Chebyshev_terms": n,
                        "Cost": float(n),
                        "Violation": 1.0 / n,
                    }
                )
    p = DataPlotter(pd.DataFrame(rows), str(tmp_path))
    p.set_plot_variables("Cost", "Violation")
    return p


def _plot(plotter, **kwargs):
    plotter.plot(
        grouping_variable="Overlap_operator_method",
        excluded_from_grouping_list=["Number_of_Chebyshev_terms"],
        file_format="png",
        verbose=False,
        **kwargs,
    )
    return sorted(path.split("/")[-1] for path in plotter._last_plot_paths.values())


def test_no_prefix_by_default(plotter):
    filenames = _plot(plotter)
    assert len(filenames) == 2
    assert all(name.startswith("Violation_Vs_Cost") for name in filenames)


def test_prefix_is_prepended(plotter):
    filenames = _plot(plotter, output_prefix="Pade")
    assert len(filenames) == 2
    assert all(name.startswith("Pade_Violation_Vs_Cost") for name in filenames)


def test_prefix_names_the_top_level_folder(plotter, tmp_path):
    _plot(plotter, output_prefix="Pade")
    for path in plotter._last_plot_paths.values():
        relative = os.path.relpath(path, tmp_path).split(os.sep)
        assert relative[:2] == [
            "Pade_Violation_Vs_Cost",
            "Grouped_by_Overlap_operator_method",
        ]


def test_no_prefix_keeps_the_plain_folder(plotter, tmp_path):
    _plot(plotter)
    for path in plotter._last_plot_paths.values():
        assert os.path.relpath(path, tmp_path).split(os.sep)[0] == "Violation_Vs_Cost"


def test_prefix_names_the_folder_without_grouping(plotter, tmp_path):
    plotter.plot(
        excluded_from_grouping_list=["Number_of_Chebyshev_terms"],
        output_prefix="Pade",
        file_format="png",
        verbose=False,
    )
    assert plotter._last_plot_paths
    for path in plotter._last_plot_paths.values():
        relative = os.path.relpath(path, tmp_path).split(os.sep)
        assert relative[0] == "Pade_Violation_Vs_Cost"
        assert relative[-1].startswith("Pade_")


def test_prefixed_and_plain_plots_do_not_collide(plotter, tmp_path):
    _plot(plotter)
    _plot(plotter, output_prefix="Pade")
    saved = list(tmp_path.rglob("*.png"))
    assert len(saved) == 4


def test_prefix_takes_precedence_over_combined(plotter):
    filenames = _plot(plotter, output_prefix="Pade", include_combined_prefix=True)
    assert all(name.startswith("Pade_") for name in filenames)
    assert not any("Combined" in name for name in filenames)


def test_prefix_via_per_figure_overrides(plotter):
    filenames = _plot(
        plotter,
        per_figure_overrides=lambda meta: (
            {"output_prefix": "Pade"}
            if meta.get("Configuration_label") == "0018000"
            else {}
        ),
    )
    assert filenames[0].startswith("Pade_") and "0018000" in filenames[0]
    assert filenames[1].startswith("Violation_Vs_Cost") and "0007200" in filenames[1]


# clear_existing is applied on the first save, to the folder actually
# written to, so a prefixed plot cannot clear (or create) the folder of
# the unprefixed plots of the same variables


def test_clear_existing_for_prefixed_plot_keeps_plain_plots(plotter, tmp_path):
    _plot(plotter)
    plotter.set_plot_variables("Cost", "Violation", clear_existing=True)
    _plot(plotter, output_prefix="Pade")
    assert len(list((tmp_path / "Violation_Vs_Cost").rglob("*.png"))) == 2
    assert len(list((tmp_path / "Pade_Violation_Vs_Cost").rglob("*.png"))) == 2


def test_prefixed_plot_does_not_create_plain_folder(plotter, tmp_path):
    plotter.set_plot_variables("Cost", "Violation", clear_existing=True)
    _plot(plotter, output_prefix="Pade")
    assert not (tmp_path / "Violation_Vs_Cost").exists()


def test_clear_existing_clears_the_folder_written_to(plotter, tmp_path):
    stale = tmp_path / "Pade_Violation_Vs_Cost" / "stale.png"
    stale.parent.mkdir()
    stale.touch()
    plotter.set_plot_variables("Cost", "Violation", clear_existing=True)
    _plot(plotter, output_prefix="Pade")
    assert not stale.exists()
    assert len(list((tmp_path / "Pade_Violation_Vs_Cost").rglob("*.png"))) == 2


def test_clear_existing_applies_once_per_set_plot_variables(plotter, tmp_path):
    plotter.set_plot_variables("Cost", "Violation", clear_existing=True)
    _plot(plotter)
    # A second plot() with another grouping must not clear the first one
    plotter.plot(
        grouping_variable="Configuration_label",
        excluded_from_grouping_list=["Number_of_Chebyshev_terms"],
        file_format="png",
        verbose=False,
    )
    assert len(list((tmp_path / "Violation_Vs_Cost").rglob("*.png"))) == 4
