"""
Unit tests for the ``excluded_from_grouping_list`` faceting option of
``DataPlotter.plot_heatmap()``.

``plot_heatmap()`` facets the data by every multivalued tunable
parameter EXCEPT the axis variables, producing one heatmap figure per
group. ``excluded_from_grouping_list`` (mirroring ``plot()``'s parameter
of the same name) lets callers drop *additional* parameters from that
faceting: the excluded parameter's values collapse onto shared (x, y)
cells and are combined via the ``aggregation`` aggfunc, so a single
heatmap is produced instead of one per value.

These tests run the real ``DataPlotter`` end-to-end (with the ``Agg``
matplotlib backend) against ``tmp_path`` and count the saved figures,
following the pattern of ``test_data_plotter_per_figure_overrides.py``.
"""

import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing

from library.visualization.plotters.data_plotter import DataPlotter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def heatmap_dataframe():
    """
    DataFrame whose only non-axis multivalued tunable parameter is
    ``Kernel_operator_type``, so the default heatmap faceting yields one
    figure per kernel.

    Columns (all faceting-eligible columns are valid multivalued tunable
    parameters):
        - Bare_mass: heatmap x-axis (3 values).
        - KL_diagonal_order: heatmap y-axis (3 values).
        - Plateau_PCAC_mass: heatmap value variable (numeric).
        - Kernel_operator_type: extra facet parameter (Wilson,
          Brillouin); produces 2 heatmaps by default.

    Total: 2 kernels x 3 bare masses x 3 orders = 18 rows.
    """
    rows = []
    for kernel in ["Wilson", "Brillouin"]:
        for bare_mass in [0.01, 0.02, 0.03]:
            for order in [1, 2, 3]:
                rows.append(
                    {
                        "Kernel_operator_type": kernel,
                        "Bare_mass": bare_mass,
                        "KL_diagonal_order": order,
                        "Plateau_PCAC_mass": bare_mass * (1 + 0.1 * order)
                        + (0.0 if kernel == "Wilson" else 0.005),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def heatmap_plotter(heatmap_dataframe, tmp_path):
    """A DataPlotter ready to draw Plateau_PCAC_mass over a
    Bare_mass x KL_diagonal_order grid."""
    p = DataPlotter(heatmap_dataframe, str(tmp_path))
    p.set_heatmap_variables("Bare_mass", "KL_diagonal_order", "Plateau_PCAC_mass")
    return p


def _saved_files(tmp_path):
    """Return a sorted list of saved figure file paths under tmp_path."""
    return sorted(p for p in tmp_path.rglob("*") if p.is_file())


# =============================================================================
# Tests
# =============================================================================


def test_baseline_facets_one_heatmap_per_kernel(heatmap_plotter, tmp_path):
    """Without exclusion, Kernel_operator_type facets into two heatmaps."""
    result = heatmap_plotter.plot_heatmap(
        save_figure=True,
        file_format="pdf",
        verbose=False,
    )

    # Method chaining works
    assert result is heatmap_plotter

    files = _saved_files(tmp_path)
    assert len(files) == 2
    for f in files:
        assert f.suffix == ".pdf"
        assert f.stat().st_size > 0


def test_excluded_parameter_collapses_to_single_heatmap(heatmap_plotter, tmp_path):
    """Excluding Kernel_operator_type folds both kernels into one heatmap."""
    heatmap_plotter.plot_heatmap(
        excluded_from_grouping_list=["Kernel_operator_type"],
        save_figure=True,
        file_format="pdf",
        verbose=False,
    )

    files = _saved_files(tmp_path)
    assert len(files) == 1
    assert files[0].suffix == ".pdf"
    assert files[0].stat().st_size > 0


def test_invalid_excluded_name_raises_value_error(heatmap_plotter):
    """A name that is not a multivalued tunable parameter is rejected."""
    with pytest.raises(ValueError):
        heatmap_plotter.plot_heatmap(
            excluded_from_grouping_list=["Not_a_parameter"],
            save_figure=False,
            verbose=False,
        )
