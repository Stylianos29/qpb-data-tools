"""Tests for the value formatting options of generate_grouped_summary_tables."""

import pandas as pd
import pytest

from library.data.table_generator import TableGenerator


def _gvar_test_generator():
    """Two configurations per kernel type, values as (mean, error)."""
    dataframe = pd.DataFrame(
        {
            "Kernel_operator_type": ["Brillouin", "Brillouin", "Wilson", "Wilson"],
            "Configuration_label": ["0000200", "0000400", "0000200", "0000400"],
            "Average_forward_operator_norm_squared_values": [
                (3.8228, 0.0001),
                (3.8231, 0.0002),
                (5.0, 0.0),  # zero error: plain value, no "(0)"
                (4.957592, 0.975653),
            ],
        }
    )
    return TableGenerator(dataframe)


def test_grouped_summary_list_with_gvar_formats_each_value():
    table = _gvar_test_generator().generate_grouped_summary_tables(
        value_variable="Average_forward_operator_norm_squared_values",
        aggregation="list",
        format_value="gvar",
        row_variable="Kernel_operator_type",
        exclude_from_grouping=["Configuration_label"],
    )

    assert "Brillouin | 3.82280(10), 3.82310(20)" in table
    assert "Wilson | 4.96(98), 5.0000" in table
    assert "(3.8228, 0.0001)" not in table


def test_grouped_summary_callable_format_value_keeps_receiving_joined_string():
    received = []

    def spy(value):
        received.append(value)
        return str(value)

    _gvar_test_generator().generate_grouped_summary_tables(
        value_variable="Average_forward_operator_norm_squared_values",
        aggregation="list",
        format_value=spy,
        row_variable="Kernel_operator_type",
        exclude_from_grouping=["Configuration_label"],
    )

    assert "(3.8228, 0.0001), (3.8231, 0.0002)" in received


def test_grouped_summary_rejects_unknown_format_value_name():
    with pytest.raises(ValueError, match="Unsupported format_value"):
        _gvar_test_generator().generate_grouped_summary_tables(
            value_variable="Average_forward_operator_norm_squared_values",
            aggregation="list",
            format_value="not_a_formatter",
        )
