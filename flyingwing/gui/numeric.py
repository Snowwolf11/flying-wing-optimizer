"""Locale-robust numeric field parsing.

`dcc.Input(type="number", ...)` renders a native `<input type="number">`,
whose text parsing/validity checking is done entirely by the browser before
Dash ever sees it -- and browsers vary by locale in which decimal separator
they accept. On a comma-locale browser, typing "." is what gets silently
rejected (the DOM reports an empty value), not ",". There is no reliable way
to force one behavior from Python: the browser has already decided the field
is invalid before any callback runs.

The fix used throughout this GUI: render these as `type="text"` instead (so
the browser never does its own numeric validity check at all) and parse the
raw string here, accepting both '.' and ',' as the decimal separator.
"""
from __future__ import annotations


def parse_number(value) -> float | None:
    """Parse a number from a dcc.Input(type="text") field's raw value,
    accepting both '.' and ',' as the decimal separator. Returns None if the
    field is empty, whitespace, or not a valid number -- callers should
    treat None as "the user hasn't entered a valid value here yet"."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None
