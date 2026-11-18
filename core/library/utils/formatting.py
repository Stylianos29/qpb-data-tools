"""
Value formatting helpers shared across the library.

Renders numerical values with their uncertainty in compact
``value(error)`` notation, e.g. ``4.957592 ± 0.975653`` as ``4.96(98)``.
Used both for plot legends and for summary tables.
"""

import math
from typing import Optional

import gvar as gv


__all__ = ["format_value_with_error"]


def _is_usable_error(error: Optional[float]) -> bool:
    """
    Check whether an uncertainty is finite, positive and thus printable.

    gvar renders a zero or non-finite uncertainty as ``4.9576(0)`` or
    ``4.9576 ± nan``, neither of which belongs in a legend, so such
    values fall back to plain numeric formatting instead.
    """
    if error is None:
        return False
    try:
        error = float(error)
    except (TypeError, ValueError):
        return False
    return math.isfinite(error) and error > 0.0


def format_value_with_error(
    mean: float,
    error: Optional[float] = None,
    fallback_format: str = ".4f",
) -> str:
    """
    Format a value in compact ``value(error)`` notation.

    Parameters:
    -----------
    mean : float
        Central value.
    error : float, optional
        Uncertainty on the central value. When omitted, zero or
        non-finite, the value is rendered with `fallback_format` and no
        uncertainty.
    fallback_format : str, optional
        Format string used when no usable uncertainty is available.
        Default is ".4f".

    Returns:
    --------
    str
        E.g. ``"4.96(98)"``, ``"-0.0048(31)"``, ``"0.8445(12)"``, or
        ``"4.9576"`` when no uncertainty is available.

    Notes:
    ------
    The number of displayed digits is chosen by gvar from the magnitude
    of the uncertainty, so a precisely determined parameter keeps its
    significant digits while a poorly determined one is not reported to
    a precision it does not have.
    """
    if not math.isfinite(float(mean)):
        return f"{float(mean):{fallback_format}}"

    if not _is_usable_error(error):
        return f"{float(mean):{fallback_format}}"

    assert error is not None
    return str(gv.gvar(float(mean), float(error)))
