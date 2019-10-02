from __future__ import annotations  # because of cyclic import `ParameterRange`
import typing
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
if typing.TYPE_CHECKING:
    from calix.common.base import ParameterRange  # pylint: disable=cyclic-import


@dataclass
class BoundaryCheckResult:
    """
    Result object for boundary checks.
    Contains:
        * Parameters clipped to a desired parameter range.
        * Boolean mask of parameters which initially exceeded the desired
        parameter range. Values of True indicate reaching the boundaries.
        * List of strings containing error messages. May be empty, as
        messages are only returned if errors are given.
    """

    parameters: np.ndarray
    error_mask: np.ndarray
    messages: List[str]


def check_range_boundaries(parameters: np.ndarray,
                           boundaries: ParameterRange,
                           errors: Optional[List[str]] = None
                           ) -> BoundaryCheckResult:
    """
    Checks if parameters are in a given range and sets them to the limits if
    they exceed the range.
    If error messages are given, returns those if a parameter has reached the
    limits or has gone beyond.

    :param parameters: Array of parameters to check.
    :param boundaries: Parameter range to be checked for.
    :param errors: List of error messages to print when parameter exceeds
        respective boundaries, in format [err_lower, err_upper]. Message needs
        to include placeholer '{0}' where indices of the violating parameters
        are inserted.

    :return: BoundaryCheckResult, containing parameters within boundaries,
        error mask and optional error messages.
    """

    result = BoundaryCheckResult(
        parameters, error_mask=np.empty_like(parameters, dtype=bool),
        messages=list())

    below_mask = parameters <= boundaries.lower
    result.parameters[below_mask] = boundaries.lower
    if errors is not None:
        if np.any(below_mask):
            result.messages.append(
                errors[0].format(below_mask.nonzero()[0]))

    above_mask = parameters >= boundaries.upper
    result.parameters[above_mask] = boundaries.upper
    if errors is not None:
        if np.any(above_mask):
            result.messages.append(
                errors[1].format(above_mask.nonzero()[0]))

    result.error_mask = below_mask | above_mask

    return result
