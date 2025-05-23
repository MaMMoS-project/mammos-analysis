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


def _check_monotonicity(arr: np.ndarray, direction=None) -> None:
    """Check if the array is monotonically increasing or decreasing.

    Args:
        arr: Input 1D numpy array.
        direction: Direction to check for monotonicity.
            Can be "increasing", "decreasing", or None (for any monotonicity).

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
    if direction == "increasing":
        if not np.all(np.diff(arr) >= 0):
            raise ValueError("Array is not monotonically increasing.")
    elif direction == "decreasing":
        if not np.all(np.diff(arr) <= 0):
            raise ValueError("Array is not monotonically decreasing.")
    else:
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
    h_val = H.value if (is_entity or is_quantity) else H
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
        return hc_val * H.unit
    else:
        return np.array(hc_val)


def extract_remanent_magnetization(
    H: me.Entity | u.Quantity | np.ndarray, M: me.Entity | u.Quantity | np.ndarray
) -> me.Entity | u.Quantity | np.ndarray:
    """Extract remanent magnetization from hysteresis loop.

    Args:
        H: External magnetic field. Can be Entity, Quantity, or numpy array.
        M: Spontaneous magnetisation. Can be Entity, Quantity, or numpy array.

    Returns:
        Remanent magnetization in the same type as the input M.

    Raises:
        ValueError: If the field does not cross the zero axis or calculation fails.
    """
    # Determine input types
    is_entity = isinstance(M, me.Entity)
    is_quantity = isinstance(M, u.Quantity)

    # Extract values for computation
    h_val = H.value if (isinstance(H, (me.Entity, u.Quantity))) else H
    m_val = M.value if (is_entity or is_quantity) else M

    # Check monotonicity on the values
    _check_monotonicity(h_val)

    # Check if field crosses zero axis
    if not ((h_val.min() <= 0) and (h_val.max() >= 0)):
        raise ValueError(
            "Field does not cross zero axis. Cannot calculate remanent magnetization."
        )

    # Interpolation only works on increasing data
    idx = np.argsort(h_val)
    h_sorted = h_val[idx]
    m_sorted = m_val[idx]

    mr_val = abs(
        np.interp(
            0.0,
            h_sorted,
            m_sorted,
            left=np.nan,
            right=np.nan,
        )
    )

    # Check if remanent magnetization is valid
    if np.isnan(mr_val):
        raise ValueError("Failed to calculate remanent magnetization.")

    # Return in the same type as input
    if is_entity:
        return me.Mr(mr_val)
    elif is_quantity:
        return mr_val * M.unit
    else:
        return np.array(mr_val)


def extract_B_curve(
    H: me.Entity | u.Quantity | np.ndarray,
    M: me.Entity | u.Quantity | np.ndarray,
    demagnetisation_coefficient: float,
) -> me.Entity:
    """Extract BH curve from hysteresis loop.

    Args:
        H: External magnetic field. Can be Entity, Quantity, or numpy array.
        M: Spontaneous magnetisation. Can be Entity, Quantity, or numpy array.
        demagnetisation_coefficient: Demagnetisation coefficient necessary
            to evaluate BHmax. If set to None, BHmax will also be None.

    Returns:
        B: Magnetic flux density as an Entity.

    Raises:
        ValueError: If the field does not cross the zero axis or calculation fails.
    """
    if isinstance(demagnetisation_coefficient, (int, float)):
        if demagnetisation_coefficient < 0 or demagnetisation_coefficient > 1:
            raise ValueError("Demagnetisation coefficient must be between 0 and 1.")
    else:
        raise ValueError("Demagnetisation coefficient must be a float or int.")
    # Convert raw numpy arrays to quantities if needed
    if isinstance(H, np.ndarray) and not isinstance(H, (me.Entity, u.Quantity)):
        H = H * u.A / u.m
    if isinstance(M, np.ndarray) and not isinstance(M, (me.Entity, u.Quantity)):
        M = M * u.A / u.m

    # Calculate internal field and flux density
    H_internal = H - demagnetisation_coefficient * M
    B_internal = (H_internal + M) * u.constants.mu0

    return me.Entity("MagneticFluxDensity", value=B_internal)


def extract_maximum_energy_product(
    H: me.Entity | u.Quantity | np.ndarray,
    B: me.Entity | u.Quantity | np.ndarray,
) -> me.Entity:
    """Extract maximum energy product from hysteresis loop.

    Args:
        H: External magnetic field. Can be Entity, Quantity, or numpy array.
        B: Magnetic flux density. Can be Entity, Quantity, or numpy array.

    Returns:
        BHmax: Maximum energy product as an Entity.

    """
    # Convert raw numpy arrays to quantities if needed
    if isinstance(H, np.ndarray) and not isinstance(H, (me.Entity, u.Quantity)):
        H = H * u.A / u.m
    else:
        H = H.to(u.A / u.m)
    if isinstance(B, np.ndarray) and not isinstance(B, (me.Entity, u.Quantity)):
        B = B * u.T
    else:
        B = B.to(u.T)

    _check_monotonicity(H.value)
    _check_monotonicity(B.value)

    # check if H is increasing or decreasing
    if np.all(np.diff(H.value) >= 0):
        # H is increasing
        H = H.value
        B = B.value
    else:
        # H is decreasing
        H = H.value[::-1]
        B = B.value[::-1]

    # if B is decreasing whilst H is increasing error
    if np.all(np.diff(B) <= 0):
        raise ValueError("B is decreasing while H is increasing, not sure what to do.")

    # Calculate maximum energy product and the applied field it occurs at
    BH = H * B
    BHmax = abs(np.min(BH))
    H_opt = H[np.argmin(BH)]

    return me.H(H_opt), me.BHmax(BHmax)


def extrinsic_properties(
    H: mammos_entity.Entity | u.Quantity | np.ndarray,
    M: mammos_entity.Entity | u.Quantity | np.ndarray,
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
    Hc = extract_coercive_field(H, M)
    Mr = extract_remanent_magnetization(H, M)

    if demagnetisation_coefficient is None:
        BHmax = me.BHmax(np.nan)
    else:
        _, BHmax = extract_maximum_energy_product(
            H, extract_B_curve(H, M, demagnetisation_coefficient)
        )
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
