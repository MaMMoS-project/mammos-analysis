"""Hysteresis analysis and postprocessing functions."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mammos_entity

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from scipy.optimize import minimize

import mammos_entity as me
import mammos_units as u


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class ExtrinsicProperties:
    """Extrinsic properties extracted from hysteresis loop.

    Args:
        Hc: Coercive field.
        Mr: Remanent magnetization.
        BHmax: Energy product.

    """

    Hc: me.Entity
    Mr: me.Entity
    BHmax: me.Entity


def _check_monotonicity(arr: np.ndarray) -> None:
    """Check if the array is monotonically increasing or decreasing.

    Args:
        arr: Input 1D numpy array.

    Raises:
        ValueError: If the array is not monotonic or contains NaN values.
    """
    # Check for NaN values
    if np.isnan(arr).any():
        raise ValueError("Array contains NaN values.")

    # Arrays with 0 or 1 elements are considered monotonic
    if arr.size <= 1:
        return

    # Check if array is monotonically increasing or decreasing
    if not (np.all(np.diff(arr) >= 0) or np.all(np.diff(arr) <= 0)):
        raise ValueError("Array is not monotonic.")


def extract_coercive_field(
    H: me.Entity | u.Quantity | np.ndarray, M: me.Entity | u.Quantity | np.ndarray
) -> me.Entity | u.Quantity | np.ndarray:
    """Extract coercive field from hysteresis loop.

    Args:
        H: External magnetic field. Can be Entity, Quantity, or numpy array.
        M: Spontaneous magnetisation. Can be Entity, Quantity, or numpy array.

    Returns:
        Coercive field in the same type as the input H.

    """
    # Determine input types
    is_entity = isinstance(H, me.Entity)
    is_quantity = isinstance(H, u.Quantity)

    # Extract values for computation
    h_val = H.to(u.A / u.m).value if (is_entity or is_quantity) else H
    m_val = M.value if (isinstance(M, (me.Entity, u.Quantity))) else M

    # Check monotonicity on the values
    _check_monotonicity(h_val)

    # Interpolation only works on increasing data
    idx = np.argsort(m_val)
    h_sorted = h_val[idx]
    m_sorted = m_val[idx]

    hc_val = abs(
        np.interp(
            0.0,
            m_sorted,
            h_sorted,
            left=np.nan,
            right=np.nan,
        )
    )

    # Check if coercive field is valid
    if np.isnan(hc_val):
        raise ValueError("Failed to calculate coercive field.")

    # Return in the same type as input
    if is_entity:
        return me.Hc(hc_val)
    elif is_quantity:
        return hc_val * u.A / u.m
    else:
        return np.array(hc_val)


def extrinsic_properties(
    H: mammos_entity.Entity,
    M: mammos_entity.Entity,
    demagnetisation_coefficient: float | None = None,
) -> ExtrinsicProperties:
    """Evaluate extrinsic properties.

    Args:
        H: External magnetic field.
        M: Spontaneous magnetisation.
        demagnetisation_coefficient: Demagnetisation coefficient necessary
            to evaluate BHmax. If set to None, BHmax will also be None.

    Raises:
        ValueError: Failed to calculate Hc.
        ValueError: Failed to calculate Mr.
        NotImplementedError: BHmax evaluation is not yet implemented.

    Returns:
        ExtrinsicProperties: _description_
    """
    h = H
    m = M

    sign_changes_m = np.where(np.diff(np.sign(m)))[0]
    sign_changes_h = np.where(np.diff(np.sign(h)))[0]

    if len(sign_changes_m) == 0:
        raise ValueError("Failed to calculate Hc.")

    if len(sign_changes_h) == 0:
        raise ValueError("Failed to calculate Mr.")

    if len(sign_changes_m) > 2:
        raise ValueError(
            "Multiple zero crossings in magnetization. "
            "Please check the data for multiple sweeps."
        )
    if len(sign_changes_h) > 2:
        raise ValueError(
            "Multiple zero crossings in field. "
            "Please check the data for multiple sweeps."
        )

    # Coercive field
    index_before = sign_changes_m[0]
    index_after = sign_changes_m[0] + 1
    Hc = abs(
        np.interp(
            0,
            [m[index_before], m[index_after]],
            [h[index_before], h[index_after]],
        )
    )

    # Remanent magnetization
    index_before = sign_changes_h[0]
    index_after = sign_changes_h[0] + 1
    Mr = abs(
        np.interp(
            0,
            [h[index_before], h[index_after]],
            [m[index_before], m[index_after]],
        )
    )
    if demagnetisation_coefficient is None:
        BHmax = me.BHmax(np.nan)
    else:
        raise NotImplementedError("BHmax evaluation is not yet implemented.")
    return ExtrinsicProperties(
        me.Hc(Hc),
        me.Mr(Mr),
        BHmax,
    )


def linearised_segment(H: mammos_entity.Entity, M: mammos_entity.Entity):
    """Evaluate linearised segment."""
    df = pd.DataFrame({"H": H, "M": M})

    h = 0.5  # threshold_training
    mar = 0.05  # margin_to_line
    m0 = 1.0  # m_guess
    i0 = 0  # index_adjustment
    try:
        upper_index = i0 + np.argmin(np.abs(df["M"] - h))
        hh_u = df["H"].iloc[upper_index]
        df_ = df[df["H"] < hh_u]

        lower_index = i0 + np.argmin(np.abs(df["M"] >= 0))
        hh_l = df["H"].iloc[lower_index]
        df_ = df_[df_["H"] >= hh_l]
    except Exception as e:
        print(f"[ERROR]: Exception: {e}")
        raise ValueError("Failed Extraction")

    if df_.shape[0] < 10:
        print(
            f"[ERROR]: Less than 10 points in margin [0,{h}] for linear regression (only {df_.shape[0]})"
        )
        return 0.0

    def line(x, m, b=0.0):
        return m * x + b

    def penalty_function(m, x, y, b=0.0):
        return np.sum((y - line(x=x, m=m, b=b)) ** 2)

    try:
        b = df_["M"].iloc[np.argmin(np.abs(df_["H"]))]
        res = minimize(
            penalty_function,
            m0,
            args=(df_["H"], df_["M"], b),
        )
        m_opt = res.x
        if not res.success:
            print(f"Optimization did not converge in general: {res.message}")
            raise ValueError("Failed Linearization")
        if m_opt > 1000 or m_opt < 0:
            print(f"[ERROR]: Slope is unreasonable: {res.x}")
            raise ValueError("Failed Linearization")
    except Exception as e:
        print(f"[ERROR]: Something did not work: {e}.")
        raise ValueError("Failed Linearization")

    try:
        margin = np.abs(df["M"] - line(df["H"], m_opt, b)) < mar
        # npo = np.sum(margin)  # number_points_in_margin
        x_max_lin = np.max(df["H"][margin])
    except Exception as e:
        print(f"[ERROR]: Failed x_max_lin extraction: {e}.")

    return x_max_lin * u.A / u.m
