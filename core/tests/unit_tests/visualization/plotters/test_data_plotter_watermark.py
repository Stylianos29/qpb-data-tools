"""
Unit tests for the watermark feature of ``DataPlotter.plot()``.

The watermark stamps a status message (typically "PRELIMINARY") across
the plotting area of each saved figure, so that results shown before
they are final cannot be mistaken for finished ones.

The tests are organized around four behavioral contracts:

1. Opt-in - no watermark is drawn unless ``watermark_text`` is given,
   so existing plotting code is unaffected.
2. Placement and styling - ``watermark_position`` and
   ``watermark_style`` reach the annotation manager and the resulting
   text object.
3. Per-figure overrides - the watermark kwargs participate in
   ``per_figure_overrides`` like any other overrideable kwarg.
4. Non-interference - the watermark stays out of the legend and is not
   applied to insets.

As with the per-figure-override tests, these run the real
``DataPlotter`` end-to-end against the ``Agg`` backend rather than
mocking the loop body.
"""

import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing
import matplotlib.pyplot as plt

from library.visualization.plotters.data_plotter import DataPlotter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def two_kernel_dataframe():
    """
    DataFrame with a two-figure outer grouping and a two-curve inner
    grouping per figure.
    """
    rows = []
    for kernel in ["Wilson", "Brillouin"]:
        for order in [1, 2]:
            for bare_mass in [0.01, 0.02, 0.03, 0.04]:
                rows.append(
                    {
                        "Kernel_operator_type": kernel,
                        "KL_diagonal_order": order,
                        "Bare_mass": bare_mass,
                        "Plateau_PCAC_mass": bare_mass * (1 + 0.1 * order),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def plotter(two_kernel_dataframe, tmp_path):
    """A DataPlotter ready to plot Plateau_PCAC_mass vs Bare_mass."""
    p = DataPlotter(two_kernel_dataframe, str(tmp_path))
    p.set_plot_variables("Bare_mass", "Plateau_PCAC_mass")
    return p


def _watermarks(ax, text):
    """Return the text objects on `ax` whose content matches `text`."""
    return [t for t in ax.texts if t.get_text() == text]


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures left open by a test."""
    yield
    plt.close("all")


# =============================================================================
# Contract 1 - Opt-in behavior
# =============================================================================


class TestWatermarkOptIn:
    """The watermark must be strictly opt-in."""

    def test_no_watermark_by_default(self, plotter):
        """Omitting watermark_text leaves the axes untouched."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        assert len(plotter._last_plot_figures) == 2
        for _, ax, _ in plotter._last_plot_figures.values():
            assert len(ax.texts) == 0

    def test_empty_watermark_text_draws_nothing(self, plotter):
        """An empty string is treated as "no watermark"."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="",
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        for _, ax, _ in plotter._last_plot_figures.values():
            assert len(ax.texts) == 0

    def test_watermark_applied_to_every_figure(self, plotter, tmp_path):
        """Each figure from a single plot() call gets one watermark."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        assert len(plotter._last_plot_figures) == 2
        for _, ax, _ in plotter._last_plot_figures.values():
            assert len(_watermarks(ax, "PRELIMINARY")) == 1

        # The figures still make it to disk.
        files = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert len(files) == 2
        for f in files:
            assert f.stat().st_size > 0


# =============================================================================
# Contract 2 - Placement and styling
# =============================================================================


class TestWatermarkPlacementAndStyling:
    """watermark_position and watermark_style must reach the text."""

    def test_default_watermark_is_centred_in_axes_coordinates(self, plotter):
        """The default watermark sits at the centre of the axes."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        for _, ax, _ in plotter._last_plot_figures.values():
            (watermark,) = _watermarks(ax, "PRELIMINARY")
            assert watermark.get_position() == (0.5, 0.5)
            assert watermark.get_transform() is ax.transAxes

    def test_watermark_position_preset(self, plotter):
        """A preset name places the watermark away from the centre."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            watermark_position="lower right",
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        expected = plotter.annotation_manager.watermark_position_presets["lower right"]
        for _, ax, _ in plotter._last_plot_figures.values():
            (watermark,) = _watermarks(ax, "PRELIMINARY")
            assert watermark.get_position() == expected

    def test_watermark_position_tuple(self, plotter):
        """Explicit axes coordinates are used verbatim."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            watermark_position=(0.3, 0.7),
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        for _, ax, _ in plotter._last_plot_figures.values():
            (watermark,) = _watermarks(ax, "PRELIMINARY")
            assert watermark.get_position() == (0.3, 0.7)

    def test_watermark_style_overrides(self, plotter):
        """watermark_style overrides the manager's defaults."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="DRAFT",
            watermark_style={
                "fontsize": 20,
                "color": "red",
                "alpha": 0.4,
                "rotation": 0,
            },
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        for _, ax, _ in plotter._last_plot_figures.values():
            (watermark,) = _watermarks(ax, "DRAFT")
            assert watermark.get_fontsize() == 20
            assert watermark.get_color() == "red"
            assert watermark.get_alpha() == 0.4
            assert watermark.get_rotation() == 0

    def test_auto_sized_watermark_fits_inside_the_axes(self, plotter):
        """The default "auto" size keeps the text within the axes."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        for fig, ax, _ in plotter._last_plot_figures.values():
            (watermark,) = _watermarks(ax, "PRELIMINARY")
            renderer = fig.canvas.get_renderer()
            text_box = watermark.get_window_extent(renderer)
            axes_box = ax.get_window_extent(renderer)
            assert text_box.width <= axes_box.width
            assert text_box.height <= axes_box.height


# =============================================================================
# Contract 3 - Per-figure overrides
# =============================================================================


class TestWatermarkPerFigureOverrides:
    """Watermark kwargs are overrideable per figure."""

    def test_watermark_text_can_vary_per_figure(self, plotter):
        """Only the figures the callback marks get a watermark."""

        def only_brillouin(metadata):
            if metadata.get("Kernel_operator_type") == "Brillouin":
                return {"watermark_text": "PRELIMINARY"}
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=only_brillouin,
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        stamped = {
            group_keys
            for group_keys, (_, ax, _) in plotter._last_plot_figures.items()
            if _watermarks(ax, "PRELIMINARY")
        }
        assert len(plotter._last_plot_figures) == 2
        assert len(stamped) == 1

    def test_watermark_style_can_vary_per_figure(self, plotter):
        """Per-figure style overrides are layered over plot()-level
        ones."""

        def red_for_brillouin(metadata):
            if metadata.get("Kernel_operator_type") == "Brillouin":
                return {"watermark_style": {"fontsize": 20, "color": "red"}}
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            watermark_style={"fontsize": 20, "color": "gray"},
            per_figure_overrides=red_for_brillouin,
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        colors = set()
        for _, ax, _ in plotter._last_plot_figures.values():
            (watermark,) = _watermarks(ax, "PRELIMINARY")
            colors.add(watermark.get_color())
        assert colors == {"gray", "red"}


# =============================================================================
# Contract 4 - Non-interference
# =============================================================================


class TestWatermarkNonInterference:
    """The watermark must not disturb the rest of the figure."""

    def test_watermark_stays_out_of_the_legend(self, plotter):
        """The watermark is not picked up as a legend entry."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            include_legend=True,
            include_plot_title=False,
            verbose=False,
        )

        for _, ax, _ in plotter._last_plot_figures.values():
            legend = ax.get_legend()
            assert legend is not None
            labels = [text.get_text() for text in legend.get_texts()]
            assert "PRELIMINARY" not in labels

    def test_watermark_is_drawn_above_the_data(self, plotter):
        """The watermark's zorder is above the plotted series."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        for _, ax, _ in plotter._last_plot_figures.values():
            (watermark,) = _watermarks(ax, "PRELIMINARY")
            data_zorders = [line.get_zorder() for line in ax.lines]
            assert all(watermark.get_zorder() > z for z in data_zorders)

    def test_insets_are_not_watermarked(self, plotter, two_kernel_dataframe, tmp_path):
        """is_inset suppresses the watermark on the inset axes."""
        fig, ax = plt.subplots()
        inset_ax = ax.inset_axes((0.5, 0.5, 0.4, 0.4))

        inset_plotter = DataPlotter(two_kernel_dataframe, str(tmp_path))
        inset_plotter.set_plot_variables("Bare_mass", "Plateau_PCAC_mass")
        inset_plotter.plot(
            grouping_variable="KL_diagonal_order",
            watermark_text="PRELIMINARY",
            target_ax=inset_ax,
            is_inset=True,
            save_figure=False,
            include_legend=False,
            include_plot_title=False,
            verbose=False,
        )

        assert _watermarks(inset_ax, "PRELIMINARY") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
