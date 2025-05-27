"""Hysteresis analysis and postprocessing functions."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numbers

if TYPE_CHECKING:
    import mammos_entity
    import mammos_units

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import mammos_entity as me
import mammos_units as u


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class ExtrinsicProperties:
    """Extrinsic properties extracted from a hysteresis loop.

    Attributes:
        Hc: Coercive field.
        Mr: Remanent magnetization.
        BHmax: Maximum energy product.
    """

    Hc: me.Entity
    Mr: me.Entity
    BHmax: me.Entity


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class MaximumEnergyProductProperties:
    """Properties related to the maximum energy product in a hysteresis loop.

    Attributes:
        Hd: Field strength at which BHmax occurs.
        Bd: Flux density at which BHmax occurs.
        BHmax: Maximum energy product value.
    """

    Hd: me.Entity
    Bd: me.Entity
    BHmax: me.Entity


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class LinearSegmentProperties:
    """Linear segment properties extracted from a hysteresis loop.

    Attributes:
        Mr: Remanent magnetization at zero field.
        Hmax: Maximum field strength in the linear segment.
        gradient: Gradient of the linear segment.
    """

    Mr: me.Entity
    Hmax: me.Entity
    gradient: u.Quantity


def _check_monotonicity(arr: np.ndarray, direction=None) -> None:
    """Check if an array is monotonically increasing or decreasing.

    Args:
        arr: Input 1D numpy array.
        direction: "increasing", "decreasing", or None for either.

    Raises:
        ValueError: If array contains NaNs or is not monotonic.
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


def _unit_processing(
    i: mammos_entity.Entity | mammos_units.Quantity | np.ndarray | numbers.Number,
    unit: mammos_units.Unit,
    return_quantity: bool = True,
) -> np.ndarray:
    """Convert input data to a consistent unit for calculations.

    Args:
        i: Input data as an Entity, Quantity, array, or number.
        unit: Target unit for conversion.
        return_quantity: If True, return a Quantity object.

    Returns:
        Data in the specified unit as a Quantity or numpy array.

    Raises:
        ValueError: If units are incompatible.
        TypeError: If input type is unsupported.
    """
    if isinstance(i, (me.Entity, u.Quantity)) and not unit.is_equivalent(i.unit):
        raise ValueError(f"Input unit {i.unit} is not equivalent to {unit}.")
    if isinstance(i, (me.Entity, u.Quantity)):
        value = i.to(unit).value
    elif isinstance(i, (np.ndarray, numbers.Number)):
        value = i
    else:
        raise TypeError(
            f"Input must be an Entity, Quantity, or numpy array, not {type(i)}."
        )

    if return_quantity:
        return u.Quantity(value, unit)
    else:
        return value


def extract_coercive_field(
    H: me.Entity | u.Quantity | np.ndarray, M: me.Entity | u.Quantity | np.ndarray
) -> me.Entity:
    """Extract the coercive field from a hysteresis loop.

    Args:
        H: External magnetic field.
        M: Spontaneous magnetization.

    Returns:
        Coercive field in the same format as H.

    Raises:
        ValueError: If the coercive field cannot be calculated.
    """
    # Extract values for computation
    h_val = _unit_processing(H, u.A / u.m)
    m_val = _unit_processing(M, u.A / u.m)

    # Check monotonicity on the values
    _check_monotonicity(h_val)

    if np.isnan(m_val).any():
        raise ValueError("Magnetization contains NaN values.")

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

    return me.Hc(hc_val)


def extract_remanent_magnetization(
    H: me.Entity | u.Quantity | np.ndarray, M: me.Entity | u.Quantity | np.ndarray
) -> me.Entity:
    """Extract the remanent magnetization from a hysteresis loop.

    Args:
        H: External magnetic field.
        M: Spontaneous magnetization.

    Returns:
        Remanent magnetization in the same format as M.

    Raises:
        ValueError: If the field does not cross zero or calculation fails.
    """
    # Determine input types
    h_val = _unit_processing(H, u.A / u.m)
    m_val = _unit_processing(M, u.A / u.m)

    # Check monotonicity on the values
    _check_monotonicity(h_val)

    if np.isnan(m_val).any():
        raise ValueError("Magnetization contains NaN values.")

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
    return me.Mr(mr_val)


def extract_B_curve(
    H: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
    M: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
    demagnetisation_coefficient: float,
) -> me.Entity:
    """Compute the B–H curve from a hysteresis loop.

    Args:
        H: External magnetic field.
        M: Spontaneous magnetization.
        demagnetisation_coefficient: Demagnetisation coefficient (0 to 1).

    Returns:
        Magnetic flux density as an Entity.

    Raises:
        ValueError: If the coefficient is out of range.
    """
    if isinstance(demagnetisation_coefficient, (int, float)):
        if demagnetisation_coefficient < 0 or demagnetisation_coefficient > 1:
            raise ValueError("Demagnetisation coefficient must be between 0 and 1.")
    else:
        raise ValueError("Demagnetisation coefficient must be a float or int.")

    H = _unit_processing(H, u.A / u.m)
    M = _unit_processing(M, u.A / u.m)

    # Calculate internal field and flux density
    H_internal = H - demagnetisation_coefficient * M
    B_internal = (H_internal + M) * u.constants.mu0

    return me.Entity("MagneticFluxDensity", value=B_internal)


def extract_maximum_energy_product(
    H: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
    B: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
) -> MaximumEnergyProductProperties:
    """Determine the maximum energy product from a hysteresis loop.

    Args:
        H: External magnetic field.
        B: Magnetic flux density.

    Returns:
        Properties of the maximum energy product.

    Raises:
        ValueError: If inputs are not monotonic or B decreases with H.
    """
    H = _unit_processing(H, u.A / u.m)
    B = _unit_processing(B, u.T)

    _check_monotonicity(H.value)
    _check_monotonicity(B.value)

    # check if H is increasing or decreasing
    if np.all(np.diff(H) >= 0):
        # H is increasing
        H = H
        B = B
    else:
        # H is decreasing
        H = H[::-1]
        B = B[::-1]

    # if B is decreasing whilst H is increasing error
    if np.all(np.diff(B) <= 0):
        raise ValueError("B is decreasing while H is increasing, not sure what to do.")

    # Calculate maximum energy product and the applied field it occurs at
    BH = H * B
    BHmax = abs(np.min(BH))
    H_d = H[np.argmin(BH)]

    # Calculate Bd, the flux density at the maximum energy product
    B_d = B[np.argmin(BH)]

    return MaximumEnergyProductProperties(
        Hd=me.H(H_d),
        Bd=me.Entity("MagneticFluxDensity", value=B_d),
        BHmax=me.BHmax(BHmax),
    )


def extrinsic_properties(
    H: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
    M: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
    demagnetisation_coefficient: float | None = None,
) -> ExtrinsicProperties:
    """Compute extrinsic properties of a hysteresis loop.

    Args:
        H: External magnetic field.
        M: Spontaneous magnetization.
        demagnetisation_coefficient: Demagnetisation coefficient for BHmax.

    Returns:
        ExtrinsicProperties containing Hc, Mr, and BHmax.

    Raises:
        ValueError: If Hc or Mr calculation fails.
    """
    Hc = extract_coercive_field(H, M)
    Mr = extract_remanent_magnetization(H, M)

    if demagnetisation_coefficient is None:
        BHmax = me.BHmax(np.nan)
    else:
        result = extract_maximum_energy_product(
            H, extract_B_curve(H, M, demagnetisation_coefficient)
        )
        BHmax = result.BHmax
    return ExtrinsicProperties(
        Hc=me.Hc(Hc),
        Mr=me.Mr(Mr),
        BHmax=BHmax,
    )


def find_linear_segment(
    H: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
    M: mammos_entity.Entity | mammos_units.Quantity | np.ndarray,
    threshold: Optional[mammos_entity.Entity | mammos_units.Quantity] = None,
    margin: Optional[mammos_entity.Entity | mammos_units.Quantity] = None,
    min_points: int = 10,
) -> LinearSegmentProperties:
    """Identify the largest field value over which the loop is linear.

    Args:
        H: Applied magnetic field values.
        M: Magnetization values.
        threshold: Upper magnetization threshold.
        margin: Allowed deviation from the linear fit.
        min_points: Minimum points required for fitting.

    Returns:
        LinearSegmentProperties with Mr, Hmax, and gradient.

    Raises:
        ValueError: For incompatible inputs or no linear region.
        RuntimeError: If slope optimization fails.
    """
    # Validate inputs
    H = _unit_processing(H, u.A / u.m)
    M = _unit_processing(M, u.A / u.m)

    if H.shape != M.shape:
        raise ValueError("H and M must have the same shape.")
    if len(H) < min_points:
        raise ValueError("Not enough data points.")

    # 1) find the index where H is closest to zero
    start = np.argmin(np.abs(H))

    last_valid = start
    # 2) grow the window
    for end in range(start + min_points - 1, len(H)):
        H_seg = H[start : end + 1]
        M_seg = M[start : end + 1]

        # 3) simple linear fit: M ≈ m*H + b
        m, b = np.polyfit(H_seg, M_seg, 1)

        # 4) compute max absolute deviation
        dev = np.abs(M_seg - (m * H_seg + b))
        if np.max(dev) <= margin:
            last_valid = end
        else:
            break

    if last_valid == start:
        raise ValueError("No linear segment found with the given parameters.")
    # 5) final fit on the maximal segment
    H_final = H[start : last_valid + 1]
    M_final = M[start : last_valid + 1]
    m_opt, b_opt = np.polyfit(H_final, M_final, 1)

    return LinearSegmentProperties(
        Mr=me.Mr(b_opt),
        Hmax=me.H(H[last_valid]),
        gradient=m_opt * u.dimensionless_unscaled,
    )
