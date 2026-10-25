"""
Unit tests for the wall-clock time and core-hours averages computed by
AnalysisCaseProcessor in _param_transform_engine.py.

The averages per spinor/vector must exclude the overhead time, which is
reported separately as Overhead_core_hours.
"""

import pandas as pd
import pytest

from src.processing._param_transform_engine import AnalysisCaseProcessor


def _inversion_dataframe(with_overhead=True):
    data = {
        "Total_calculation_time": [1000.0, 2200.0],
        "Number_of_spinors": [12, 12],
        "Number_of_vectors": [1, 2],
        "Number_of_cores": [36, 36],
    }
    if with_overhead:
        data["Total_overhead_time"] = [40.0, 100.0]
    return pd.DataFrame(data)


def test_inversion_average_excludes_overhead():
    df = AnalysisCaseProcessor(_inversion_dataframe()).process_analysis_cases()

    expected_wall_clock = pd.Series([960.0 / 12, 2100.0 / 24])
    pd.testing.assert_series_equal(
        df["Average_wall_clock_time_per_spinor"],
        expected_wall_clock,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        df["Average_core_hours_per_spinor"],
        36 * expected_wall_clock / 3600,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        df["Overhead_core_hours"],
        pd.Series([40.0 * 36 / 3600, 100.0 * 36 / 3600]),
        check_names=False,
    )


def test_inversion_average_without_overhead_column():
    df = AnalysisCaseProcessor(
        _inversion_dataframe(with_overhead=False)
    ).process_analysis_cases()

    assert df["Average_wall_clock_time_per_spinor"].tolist() == pytest.approx(
        [1000.0 / 12, 2200.0 / 24]
    )
    assert "Overhead_core_hours" not in df.columns


def test_forward_average_excludes_overhead():
    df = pd.DataFrame(
        {
            "Total_calculation_time": [500.0],
            "Total_overhead_time": [50.0],
            "Number_of_vectors": [3],
            "Number_of_cores": [72],
        }
    )
    df = AnalysisCaseProcessor(df).process_analysis_cases()

    assert df["Average_wall_clock_time_per_vector"].iloc[0] == pytest.approx(
        450.0 / 3
    )
    assert df["Average_core_hours_per_vector"].iloc[0] == pytest.approx(
        72 * 150.0 / 3600
    )
