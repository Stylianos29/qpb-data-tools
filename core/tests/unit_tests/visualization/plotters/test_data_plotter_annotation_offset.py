"""
Unit tests for the per-group form of ``annotation_offset`` in
``DataPlotter.plot()``.

``annotation_offset`` accepts either a single (dx, dy) tuple applied to
every curve, or a dict keyed by group value (the same keys as
``marker_color_map``) giving each curve its own offset. Groups missing
from the dict fall back to the default (0, 10).
"""

import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing

from library.visualization.plotters.data_plotter import DataPlotter


@pytest.fixture
def plotter(tmp_path):
    """
    DataPlotter with one figure of four curves, grouped by
    (Overlap_operator_method, Kernel_operator_type).
    """
    rows = []
    for method in ["Chebyshev", "Diagonal KL"]:
        for kernel in ["Wilson", "Brillouin"]:
            for n in [2, 4, 6]:
                rows.append(
                    {
                        "Overlap_operator_method": method,
                        "Kernel_operator_type": kernel,
                        "Number_of_Chebyshev_terms": n,
                        "Cost": float(n),
                        "Violation": 1.0 / n,
                    }
                )
    p = DataPlotter(pd.DataFrame(rows), str(tmp_path))
    p.set_plot_variables("Cost", "Violation")
    return p


def _record_offsets(plotter, monkeypatch):
    """Patch the annotation manager to record the offset used per group."""
    recorded = {}

    def fake_add(**kwargs):
        df = kwargs["dataframe"]
        key = (
            df["Overlap_operator_method"].iloc[0],
            df["Kernel_operator_type"].iloc[0],
        )
        recorded[key] = kwargs["style_overrides"]["offset"]

    monkeypatch.setattr(
        plotter.annotation_manager, "add_data_point_annotations", fake_add
    )
    return recorded


def _plot(plotter, offset):
    plotter.plot(
        grouping_variable=["Overlap_operator_method", "Kernel_operator_type"],
        excluded_from_grouping_list=["Number_of_Chebyshev_terms"],
        annotation_variable="Number_of_Chebyshev_terms",
        annotation_offset=offset,
        file_format="png",
    )


def test_tuple_offset_applies_to_all_groups(plotter, monkeypatch):
    recorded = _record_offsets(plotter, monkeypatch)
    _plot(plotter, (3, -7))
    assert len(recorded) == 4
    assert all(offset == (3, -7) for offset in recorded.values())


def test_dict_offset_is_resolved_per_group(plotter, monkeypatch):
    recorded = _record_offsets(plotter, monkeypatch)
    _plot(
        plotter,
        {
            ("Chebyshev", "Wilson"): (0, 12),
            ("Chebyshev", "Brillouin"): (0, -18),
        },
    )
    assert recorded[("Chebyshev", "Wilson")] == (0, 12)
    assert recorded[("Chebyshev", "Brillouin")] == (0, -18)
    # Groups not in the dict fall back to the default
    assert recorded[("Diagonal KL", "Wilson")] == (0, 10)
    assert recorded[("Diagonal KL", "Brillouin")] == (0, 10)


def test_dict_offset_via_per_figure_overrides(plotter, monkeypatch):
    recorded = _record_offsets(plotter, monkeypatch)
    plotter.plot(
        grouping_variable=["Overlap_operator_method", "Kernel_operator_type"],
        excluded_from_grouping_list=["Number_of_Chebyshev_terms"],
        annotation_variable="Number_of_Chebyshev_terms",
        per_figure_overrides=lambda meta: {
            "annotation_offset": {("Chebyshev", "Brillouin"): (-15, 5)}
        },
        file_format="png",
    )
    assert recorded[("Chebyshev", "Brillouin")] == (-15, 5)
    assert recorded[("Chebyshev", "Wilson")] == (0, 10)
