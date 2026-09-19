"""
Unit tests for file contents parameter extraction.

This module tests the extraction of scalar parameters from qpb log file
contents, focusing on the "Conf shifts" line that supplies the
Gauge_shifts parameter.
"""

from unittest.mock import Mock

from library.parsing.file_contents_parser import (
    extract_scalar_parameters_from_file_contents,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


def _log_header(conf_shifts_line=None):
    """
    Build a minimal qpb log header, optionally with a "Conf shifts" line.

    Args:
        - conf_shifts_line (str): Full line to insert, including its line
          terminator. If None, no "Conf shifts" line is included.

    Returns:
        List of lines as produced by file.readlines()
    """
    lines = [
        " Partition: dc\n",
        " (Lt, Lz, Ly, Lx) = (48,24,24,24)\n",
        " APE iterations = 0\n",
    ]
    if conf_shifts_line is not None:
        lines.append(conf_shifts_line)
    lines.append(" BC in time = 1\n")

    return lines


# ==============================================================================
# TESTS: SUCCESSFUL EXTRACTION
# ==============================================================================


class TestGaugeShiftsExtraction:
    """Test extraction of the Gauge_shifts parameter."""

    def test_nonzero_shifts(self):
        """Values are stored verbatim, exactly as written in the file."""
        file_contents = _log_header(" Conf shifts = 14 7 6 2\n")

        extracted_values = extract_scalar_parameters_from_file_contents(file_contents)

        assert extracted_values["Gauge_shifts"] == "14 7 6 2"

    def test_zero_shifts(self):
        """All-zero shifts are extracted rather than treated as absent."""
        file_contents = _log_header(" Conf shifts = 0 0 0 0\n")

        extracted_values = extract_scalar_parameters_from_file_contents(file_contents)

        assert extracted_values["Gauge_shifts"] == "0 0 0 0"

    def test_carriage_return_line_ending(self):
        """Windows-style line endings do not leak into the stored value."""
        file_contents = _log_header(" Conf shifts = 0 0 0 0\r\n")

        extracted_values = extract_scalar_parameters_from_file_contents(file_contents)

        assert extracted_values["Gauge_shifts"] == "0 0 0 0"

    def test_multidigit_shifts(self):
        """Shifts of more than one digit each are extracted intact."""
        file_contents = _log_header(" Conf shifts = 40 8 10 16\n")

        extracted_values = extract_scalar_parameters_from_file_contents(file_contents)

        assert extracted_values["Gauge_shifts"] == "40 8 10 16"


# ==============================================================================
# TESTS: MISSING OR MALFORMED INPUT
# ==============================================================================


class TestGaugeShiftsOmission:
    """Test that invalid "Conf shifts" lines omit the parameter."""

    def test_missing_line(self):
        """No "Conf shifts" line means no Gauge_shifts parameter."""
        file_contents = _log_header()
        logger = Mock()

        extracted_values = extract_scalar_parameters_from_file_contents(
            file_contents, logger
        )

        assert "Gauge_shifts" not in extracted_values
        # A missing line is not an error; it must not be reported as one
        assert logger.warning.call_count == 0

    def test_too_few_integers(self):
        """A line with 3 integers is omitted and a warning is logged."""
        file_contents = _log_header(" Conf shifts = 14 7 6\n")
        logger = Mock()

        extracted_values = extract_scalar_parameters_from_file_contents(
            file_contents, logger
        )

        assert "Gauge_shifts" not in extracted_values
        assert logger.warning.call_count == 1
        assert "Gauge_shifts" in logger.warning.call_args[0][0]

    def test_too_many_integers(self):
        """A line with 5 integers is omitted and a warning is logged."""
        file_contents = _log_header(" Conf shifts = 14 7 6 2 1\n")
        logger = Mock()

        extracted_values = extract_scalar_parameters_from_file_contents(
            file_contents, logger
        )

        assert "Gauge_shifts" not in extracted_values
        assert logger.warning.call_count == 1

    def test_non_integer_values(self):
        """Non-integer shift values are omitted and a warning is logged."""
        file_contents = _log_header(" Conf shifts = 1.5 7 6 2\n")
        logger = Mock()

        extracted_values = extract_scalar_parameters_from_file_contents(
            file_contents, logger
        )

        assert "Gauge_shifts" not in extracted_values
        assert logger.warning.call_count == 1
