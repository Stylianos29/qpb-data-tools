"""
Unit tests for ``statistics_location`` in ``DataPlotter.plot_histogram()``.

The summary-statistics box reuses the fit-label placement vocabulary.
Two contracts matter:

1. Every name in that vocabulary, including the matplotlib-style
   aliases "upper center" / "lower center", places the box where it
   says.
2. An unknown name raises ``ValueError`` before any figure is created,
   instead of silently falling back to the default corner.

As with the other DataPlotter tests, these run the real plotter against
the ``Agg`` backend.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing
import matplotlib.pyplot as plt

from library.visualization.plotters.data_plotter import DataPlotter


@pytest.fixture
def plotter(tmp_path):
    """A DataPlotter holding one distribution to histogram."""
    rng = np.random.default_rng(1234)
    dataframe = pd.DataFrame(
        {
            "Kernel_operator_type": ["Wilson"] * 40 + ["Brillouin"] * 40,
            "Leftmost_real_kernel_eigenvalue": rng.normal(size=80),
        }
    )
    p = DataPlotter(dataframe, str(tmp_path))
    p.set_histogram_variable("Leftmost_real_kernel_eigenvalue")
    return p


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures left open by a test."""
    yield
    plt.close("all")


def _statistics_text(plot_kwargs, plotter):
    """Plot without saving and return the statistics text object."""
    # Kernel_operator_type is multivalued, so without this exclusion the
    # data would be faceted into one figure per kernel.
    plotter.plot_histogram(
        excluded_from_grouping_list=["Kernel_operator_type"],
        save_figure=False,
        verbose=False,
        **plot_kwargs,
    )
    figures = plt.get_fignums()
    assert len(figures) == 1
    texts = [t for t in plt.figure(figures[0]).axes[0].texts if "bins =" in t.get_text()]
    assert len(texts) == 1
    return texts[0]


class TestStatisticsLocationPlacement:
    """Each accepted name puts the box at the matching spot."""

    @pytest.mark.parametrize(
        "location, position, alignment",
        [
            ("top right", (0.95, 0.95), ("right", "top")),
            ("top left", (0.05, 0.95), ("left", "top")),
            ("top center", (0.5, 0.95), ("center", "top")),
            ("upper center", (0.5, 0.95), ("center", "top")),
            ("bottom center", (0.5, 0.05), ("center", "bottom")),
            ("lower center", (0.5, 0.05), ("center", "bottom")),
            ("center", (0.5, 0.5), ("center", "center")),
        ],
    )
    def test_box_placed_at_named_location(
        self, plotter, location, position, alignment
    ):
        text = _statistics_text({"statistics_location": location}, plotter)

        assert text.get_position() == position
        assert (text.get_ha(), text.get_va()) == alignment

    def test_default_is_top_right(self, plotter):
        text = _statistics_text({}, plotter)

        assert text.get_position() == (0.95, 0.95)


class TestStatisticsLocationValidation:
    """Unknown names are rejected up front."""

    def test_unknown_location_raises(self, plotter):
        with pytest.raises(ValueError, match="statistics_location"):
            plotter.plot_histogram(
                statistics_location="middle", save_figure=False, verbose=False
            )

    def test_error_lists_valid_names(self, plotter):
        with pytest.raises(ValueError) as excinfo:
            plotter.plot_histogram(
                statistics_location="middle", save_figure=False, verbose=False
            )

        message = str(excinfo.value)
        assert "'middle'" in message
        assert "'upper center'" in message
        assert "'top right'" in message

    def test_rejected_before_any_figure_or_file(self, plotter, tmp_path):
        """Nothing is drawn or written when the location is bad."""
        with pytest.raises(ValueError):
            plotter.plot_histogram(
                grouping_variable="Kernel_operator_type",
                statistics_location="middle",
                verbose=False,
            )

        assert plt.get_fignums() == []
        assert [p for p in tmp_path.rglob("*") if p.is_file()] == []

    def test_location_unchecked_when_statistics_hidden(self, plotter):
        """With show_statistics=False the location is never used."""
        plotter.plot_histogram(
            excluded_from_grouping_list=["Kernel_operator_type"],
            show_statistics=False,
            statistics_location="middle",
            save_figure=False,
            verbose=False,
        )

        assert len(plt.get_fignums()) == 1
