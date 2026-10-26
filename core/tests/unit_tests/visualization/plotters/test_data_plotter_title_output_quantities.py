"""
Unit tests for ``title_output_quantities`` in ``DataPlotter.plot()``.

Output quantities normally vary between data points, so they may only
appear in a plot title when they are constant across the figure. The
tests cover three contracts:

1. Opt-in - without ``title_output_quantities`` the title is unchanged.
2. Constant quantities - a value that agrees to 2 decimal places across
   the figure is shown, with numerical noise tolerated.
3. Non-constant quantities - anything else raises ``ValueError``.
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


def _make_dataframe(condition_numbers):
    """
    One figure per kernel; `condition_numbers` maps kernel -> list of
    per-row condition numbers (cycled over the rows of that kernel).
    """
    rows = []
    for kernel, kappas in condition_numbers.items():
        i = 0
        for order in [1, 2]:
            for bare_mass in [0.01, 0.02, 0.03]:
                rows.append(
                    {
                        "Kernel_operator_type": kernel,
                        "KL_diagonal_order": order,
                        "Bare_mass": bare_mass,
                        "Plateau_PCAC_mass": bare_mass * (1 + 0.1 * order),
                        "Condition_number": kappas[i % len(kappas)],
                    }
                )
                i += 1
    return pd.DataFrame(rows)


def _plotter(df, tmp_path):
    p = DataPlotter(df, str(tmp_path))
    p.set_plot_variables("Bare_mass", "Plateau_PCAC_mass")
    return p


def _titles(plotter):
    return [ax.get_title() for _, ax, _ in plotter._last_plot_figures.values()]


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures left open by a test."""
    yield
    plt.close("all")


# =============================================================================
# Contract 1 - Opt-in
# =============================================================================


def test_title_unchanged_by_default(tmp_path):
    p = _plotter(_make_dataframe({"Wilson": [12.3], "Brillouin": [4.5]}), tmp_path)
    p.plot(grouping_variable="KL_diagonal_order", include_plot_title=True)

    for title in _titles(p):
        assert "kappa" not in title


# =============================================================================
# Contract 2 - Constant quantities
# =============================================================================


def test_constant_quantity_shown_per_figure(tmp_path):
    p = _plotter(_make_dataframe({"Wilson": [12.3], "Brillouin": [4.5]}), tmp_path)
    p.plot(
        grouping_variable="KL_diagonal_order",
        include_plot_title=True,
        title_output_quantities=["Condition_number"],
    )

    titles = sorted(_titles(p))
    assert any("=12.30" in t for t in titles)
    assert any("=4.50" in t for t in titles)
    assert all("kappa" in t for t in titles)


def test_numerical_noise_tolerated(tmp_path):
    # Straddles a rounding boundary (12.344999 vs 12.345001) but the
    # spread is far below half a unit in the 2nd decimal place.
    df = _make_dataframe({"Wilson": [12.344999, 12.345001]})
    p = _plotter(df, tmp_path)
    p.plot(
        grouping_variable="KL_diagonal_order",
        include_plot_title=True,
        title_output_quantities=["Condition_number"],
    )

    assert "=12.35" in _titles(p)[0] or "=12.34" in _titles(p)[0]


# =============================================================================
# Contract 3 - Non-constant quantities
# =============================================================================


def test_varying_quantity_raises(tmp_path):
    p = _plotter(_make_dataframe({"Wilson": [12.30, 12.31]}), tmp_path)
    with pytest.raises(ValueError, match="not constant"):
        p.plot(
            grouping_variable="KL_diagonal_order",
            include_plot_title=True,
            title_output_quantities=["Condition_number"],
        )


def test_missing_column_raises(tmp_path):
    p = _plotter(_make_dataframe({"Wilson": [12.3]}), tmp_path)
    with pytest.raises(ValueError, match="not a column"):
        p.plot(
            grouping_variable="KL_diagonal_order",
            include_plot_title=True,
            title_output_quantities=["Minimum_eigenvalue"],
        )


# =============================================================================
# title_from_columns integration
# =============================================================================


def test_output_quantity_in_title_from_columns(tmp_path):
    p = _plotter(_make_dataframe({"Wilson": [12.3], "Brillouin": [4.5]}), tmp_path)
    p.plot(
        grouping_variable="KL_diagonal_order",
        include_plot_title=True,
        title_from_columns=["Kernel_operator_type", "Condition_number"],
    )

    titles = sorted(_titles(p))
    assert titles[0].startswith("Brillouin Kernel, ")
    assert titles[0].endswith("=4.50")
    assert titles[1].startswith("Wilson Kernel, ")
    assert titles[1].endswith("=12.30")


def test_varying_quantity_in_title_from_columns_raises(tmp_path):
    p = _plotter(_make_dataframe({"Wilson": [12.30, 12.31]}), tmp_path)
    with pytest.raises(ValueError, match="not constant"):
        p.plot(
            grouping_variable="KL_diagonal_order",
            include_plot_title=True,
            title_from_columns=["Kernel_operator_type", "Condition_number"],
        )


def test_title_output_quantities_combined_with_title_from_columns(tmp_path):
    p = _plotter(_make_dataframe({"Wilson": [12.3]}), tmp_path)
    p.plot(
        grouping_variable="KL_diagonal_order",
        include_plot_title=True,
        title_from_columns=["Kernel_operator_type"],
        title_output_quantities=["Condition_number"],
    )

    title = _titles(p)[0]
    assert title.startswith("Wilson Kernel, ")
    assert title.endswith("=12.30")
