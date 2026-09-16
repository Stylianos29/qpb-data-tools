"""
Formatting helpers for fit-result legend labels.

Provides a single place to render fit parameters the way they should
appear in publication-quality legends:

    - Values carry their uncertainty in compact notation, so that
      ``4.957592 ± 0.975653`` is displayed as ``4.96(98)``.
    - Signs are pulled out of the value and folded into the preceding
      operator, so that no label ever reads ``+ -4.59``.

The low-level helpers (:func:`format_value_with_error` and
:func:`format_signed_term`) are deliberately equation-agnostic; the
equation builders on top of them cover the functional forms currently
fitted in the analysis pipeline.

Examples:
    >>> format_value_with_error(4.957592, 0.975653)
    '4.96(98)'
    >>> format_shifted_power_law_equation(
    ...     4.957592, -0.004826, -4.586130,
    ...     a_error=0.975653, b_error=0.003102, c_error=12.186871,
    ...     variable="m",
    ... )
    'Cost = 4.96(98)/(m + 0.0048(31)) - 5(12)'
"""

import math
from typing import Optional, Sequence

import gvar as gv


__all__ = [
    "format_value_with_error",
    "format_signed_term",
    "format_polynomial_equation",
    "format_shifted_power_law_equation",
]


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


def format_signed_term(
    mean: float,
    error: Optional[float] = None,
    leading: bool = False,
    fallback_format: str = ".4f",
) -> str:
    """
    Format a term with its sign pulled out into an explicit operator.

    This is what keeps ``+ -4.59`` out of legends: the sign becomes the
    operator joining this term to the preceding one, and the magnitude
    is printed unsigned.

    Parameters:
    -----------
    mean : float
        Central value of the term, whose sign selects the operator.
    error : float, optional
        Uncertainty on the term.
    leading : bool, optional
        When True the term starts an expression, so a positive value
        gets no operator at all and a negative one is prefixed with a
        tight ``-``. When False (default) the term is joined to what
        precedes it with a spaced ``+`` or ``-``.
    fallback_format : str, optional
        Format string used when no usable uncertainty is available.

    Returns:
    --------
    str
        E.g. ``"- 5(12)"``, ``"+ 0.0048(31)"``, or ``"-5(12)"`` when
        `leading` is True.
    """
    value = float(mean)
    negative = value < 0.0

    magnitude = format_value_with_error(
        abs(value), error, fallback_format=fallback_format
    )

    if leading:
        return f"-{magnitude}" if negative else magnitude

    operator = "-" if negative else "+"
    return f"{operator} {magnitude}"


def format_polynomial_equation(
    coefficients: Sequence[float],
    errors: Optional[Sequence[Optional[float]]] = None,
    variable: str = "m",
    lhs: Optional[str] = None,
    fallback_format: str = ".4f",
) -> str:
    """
    Format a polynomial fit as a signed, uncertainty-carrying equation.

    Coefficients are given in descending powers, so ``[slope,
    intercept]`` is a straight line and ``[a, b, c]`` is a quadratic.

    Parameters:
    -----------
    coefficients : sequence of float
        Coefficients in descending powers of `variable`.
    errors : sequence of float, optional
        Uncertainties matching `coefficients` elementwise. Entries may
        be None to suppress the uncertainty on a single coefficient.
    variable : str, optional
        Bare symbol for the independent variable, e.g. ``"m"`` or
        ``"am"`` — not pre-wrapped in ``$...$``. The function wraps each
        occurrence in mathtext itself (exponent included), so that a
        squared term renders as a true superscript instead of a literal
        ``^2``.
    lhs : str, optional
        Left-hand side; when given, the result is prefixed with
        ``"{lhs} = "``.
    fallback_format : str, optional
        Format string used when no usable uncertainty is available.

    Returns:
    --------
    str
        E.g. ``"0.8445(12) $m$ + 0.000128(53)"``.

    Raises:
    -------
    ValueError
        If `coefficients` is empty, or `errors` has a different length.
    """
    if not len(coefficients):
        raise ValueError("At least one coefficient is required")

    if errors is None:
        errors = [None] * len(coefficients)
    elif len(errors) != len(coefficients):
        raise ValueError(
            f"Got {len(errors)} errors for {len(coefficients)} coefficients"
        )

    highest_power = len(coefficients) - 1
    terms = []

    for index, (coefficient, error) in enumerate(zip(coefficients, errors)):
        power = highest_power - index

        term = format_signed_term(
            coefficient,
            error,
            leading=not terms,
            fallback_format=fallback_format,
        )

        # A leading space sets the variable off from the number, and the
        # exponent (when present) stays inside the same $...$ block so it
        # renders as a superscript rather than a literal caret.
        if power == 1:
            term = f"{term} ${variable}$"
        elif power > 1:
            term = f"{term} ${variable}^{power}$"

        terms.append(term)

    equation = " ".join(terms)

    return f"{lhs} = {equation}" if lhs else equation


def format_shifted_power_law_equation(
    a: float,
    b: float,
    c: float,
    a_error: Optional[float] = None,
    b_error: Optional[float] = None,
    c_error: Optional[float] = None,
    variable: str = "m",
    lhs: Optional[str] = "Cost",
    fallback_format: str = ".4f",
) -> str:
    """
    Format a shifted power law ``a/(x - b) + c`` as an equation.

    The shift `b` sits behind a minus sign in the functional form, so a
    negative `b` is rendered as ``(x + |b|)`` rather than ``(x - -b)``.

    Parameters:
    -----------
    a, b, c : float
        Fitted parameters of ``a/(variable - b) + c``.
    a_error, b_error, c_error : float, optional
        Uncertainties on the corresponding parameters.
    variable : str, optional
        Symbol for the independent variable, passed through verbatim so
        that callers may supply mathtext (e.g. ``"$m$"``).
    lhs : str, optional
        Left-hand side; when given, the result is prefixed with
        ``"{lhs} = "``. Default is "Cost".
    fallback_format : str, optional
        Format string used when no usable uncertainty is available.

    Returns:
    --------
    str
        E.g. ``"Cost = 4.96(98)/(m + 0.0048(31)) - 5(12)"``.
    """
    numerator = format_signed_term(
        a, a_error, leading=True, fallback_format=fallback_format
    )

    # The functional form already subtracts b, so the displayed operator
    # is the opposite of b's sign.
    shift_operator = "+" if float(b) < 0.0 else "-"
    shift_magnitude = format_value_with_error(
        abs(float(b)), b_error, fallback_format=fallback_format
    )

    offset = format_signed_term(c, c_error, fallback_format=fallback_format)

    equation = f"{numerator}/( {variable} {shift_operator} {shift_magnitude} ) {offset}"

    return f"{lhs} = {equation}" if lhs else equation
