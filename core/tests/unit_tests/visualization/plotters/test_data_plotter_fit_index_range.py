"""
Unit tests for the ``fit_index_range`` option of ``DataPlotter.plot()``.

``fit_index_range`` is a (start, stop) tuple selecting the data points
used for curve fitting. It must work for figures both with and without
a ``grouping_variable``. The tests run the real ``DataPlotter``
end-to-end against the ``Agg`` backend.
"""

import warnings

import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing
import matplotlib.pyplot as plt

from library.visualization.plotters.data_plotter import DataPlotter


def _y_value(index, bare_mass):
    """y = 2x + 1 for the first four points, far off that line after."""
    offset = 0.0 if index < 4 else 10.0 * index
    return 2.0 * bare_mass + 1.0 + offset


@pytest.fixture
def plotter(tmp_path):
    """A DataPlotter with two curves of six points each."""
    rows = [
        {
            "KL_diagonal_order": order,
            "Bare_mass": float(bare_mass),
            "Plateau_PCAC_mass": _y_value(index, bare_mass),
        }
        for order in [1, 2]
        for index, bare_mass in enumerate([1, 2, 3, 4, 5, 6])
    ]
    p = DataPlotter(pd.DataFrame(rows), str(tmp_path))
    p.set_plot_variables("Bare_mass", "Plateau_PCAC_mass")
    return p


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures left open by a test."""
    yield
    plt.close("all")


def _plot_fit_results(plotter, **kwargs):
    """Plot with a linear fit, failing on any "Fit failed" warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plotter.plot(fit_function="linear", save_figure=False, **kwargs)

    assert not [w for w in caught if "Fit failed" in str(w.message)]
    results = list(plotter.get_fit_results().values())
    assert len(results) > 0
    return results


@pytest.mark.parametrize(
    "plot_kwargs",
    [{}, {"grouping_variable": "KL_diagonal_order"}],
    ids=["individual", "grouped"],
)
class TestFitIndexRange:
    """The range applies with and without grouping."""

    def test_range_selects_fitted_points(self, plotter, plot_kwargs):
        results = _plot_fit_results(plotter, fit_index_range=(0, 4), **plot_kwargs)

        assert len(results) == 2
        for result in results:
            assert result["parameters"] == pytest.approx([2.0, 1.0])
            assert result["x_curve"][[0, -1]] == pytest.approx([1.0, 4.0])

    def test_open_ended_range(self, plotter, plot_kwargs):
        results = _plot_fit_results(plotter, fit_index_range=(4, None), **plot_kwargs)

        for result in results:
            assert result["x_curve"][[0, -1]] == pytest.approx([5.0, 6.0])

    def test_per_figure_override(self, plotter, plot_kwargs):
        results = _plot_fit_results(
            plotter,
            per_figure_overrides=lambda metadata: {"fit_index_range": (1, 4)},
            **plot_kwargs,
        )

        for result in results:
            assert result["x_curve"][[0, -1]] == pytest.approx([2.0, 4.0])

    def test_invalid_range_raises(self, plotter, plot_kwargs):
        with pytest.raises(ValueError, match="fit_index_range"):
            plotter.plot(
                fit_function="linear",
                fit_index_range=(0, 2, 1),
                save_figure=False,
                **plot_kwargs,
            )
