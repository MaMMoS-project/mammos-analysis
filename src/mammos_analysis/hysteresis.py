"""Hysteresis analysis and postprocessing functions."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numbers
import re

if TYPE_CHECKING:
    import mammos_entity
    import mammos_units
    import matplotlib

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

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
    _H: Optional[me.Entity] = None
    _M: Optional[me.Entity] = None

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Plot the spontaneous magnetization data-points."""
        if not ax:
            _, ax = plt.subplots()
        ax.scatter(self._H, y=self._M, label="Data")
        ax.axvline(self.Hmax.value, color="k", linestyle="--", label="Hmax")

        x = np.linspace(0, self.Hmax.value, 100)
        y = self.gradient.value * x + self.Mr.value
        plt.plot(x, y, linestyle="--", c="r", label="Linear fit")
        plt.legend()
        ax.set_xlabel(
            re.sub(r"(?<!^)(?=[A-Z])", " ", f"{self._H.ontology_label}")
            + f" [{self._H.unit}]"
        )
        ax.set_ylabel(
            re.sub(r"(?<!^)(?=[A-Z])", " ", f"{self._M.ontology_label}")
            + f" [{self._M.unit}]"
        )
        return ax


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
    margin: mammos_entity.Entity | mammos_units.Quantity | numbers.Number,
    method: str = "maxdev",
    min_points: int = 5,
) -> LinearSegmentProperties:
    r"""Identify the largest field value over which the hysteresis loop is linear.

    There are two possible criteria, selected by the `method` argument:

    1. **Max‐Deviation Criterion** (`method="maxdev"`):
       Require that every data point in the segment satisfies
       \[
         \max_{\,i_0 \le i \le i_{\max}}\;\bigl|\,M_i - (m\,H_i + b)\bigr|
         \;\le\; \delta,
       \]
       where:
       - \(\{(H_i, M_i)\}\) are the data points,
       - \(m\) is the fitted slope,
       - \(b\) is the fitted intercept (value of \(M\) at \(H=0\)),
       - \(\delta\) is the user‐supplied margin (in the same units as \(M\)).
       This guarantees **each** point lies within \(\pm \delta\).

    2. **RMS Criterion** (`method="rms"`):
       Require that the root‐mean‐square error over the segment satisfies
       \[
         \mathrm{RMSE}
         \;=\;
         \sqrt{\frac{1}{n}\sum_{\,i=i_0}^{\,i_{\max}}
         \bigl(M_i - (m\,H_i + b)\bigr)^2}
         \;\le\; \delta,
       \]
       where \(n = i_{\max} - i_0 + 1\).  Occasional points may exceed \(\delta\)
       provided the overall RMS error remains within \(\delta\).

    We return a single `LinearSegmentProperties` instance containing:
      - `Mr`: fitted intercept \(b\) (magnetization at \(H=0\)),
      - `Hmax`: largest field value up to which data remain “linear” under the
         chosen criterion,
      - `gradient`: fitted slope \(m\) (dimensionless).

    Parameters
    ----------
    H : Applied magnetic field values. Must be monotonic.
    M : Magnetization values corresponding to `H`.
    margin : Allowed deviation \(\delta\).
    method : Which deviation test to use:
          - `"maxdev"` (default): per‐point maximum deviation,
          - `"rms"`: root‐mean‐square deviation.
    min_points : Minimum number of points required to attempt any fit.

    Returns:
    -------
    LinearSegmentProperties
        Contains:
          - `Mr`: intercept \(b\) (magnetization at \(H=0\)),
          - `Hmax`: largest field for which the chosen criterion holds,
          - `gradient`: slope \(m\) of the fitted line on that segment.

    Notes:
    -----
    **Growing‐Window Fit**
       We attempt to extend the segment one index at a time:
       \[
         \{\,i_0,\,i_0+1,\,\dots,\,i\,\}.
       \]
       For each candidate endpoint \(i\), we fit a line
       \(\hat{M}(H) = m\,H + b\) via `np.polyfit(H[i_0:i+1], M[i_0:i+1], 1)`.
       Then we compute either:
         - **Max‐Deviation**:
           \(\max_{j=i_0}^i\,\bigl|M_j - (m\,H_j + b)\bigr| \le \delta,\) or
         - **RMS**:
            \[
             \mathrm{RMSE} \;=\;
             \sqrt{\frac{1}{\,i - i_0 + 1\,}
                   \sum_{j=i_0}^{i}
                     \bigl(M_j - (m\,H_j + b)\bigr)^2}
             \;\le\; \delta.
           \]

       As soon as adding \(i+1\) would violate the chosen inequality, we stop
       and take \(i_{\max} = i\). We then refit \((m,b)\) on \(\{i_0,\dots,i_{\max}\}\)
       to produce the final slope/intercept returned.

    """
    # 1) Normalize inputs to unitless numpy arrays in A/m
    H_arr = _unit_processing(H, u.A / u.m, return_quantity=False)
    M_arr = _unit_processing(M, u.A / u.m, return_quantity=False)
    margin_val = _unit_processing(margin, u.A / u.m, return_quantity=False)

    # 2) Basic sanity checks
    if H_arr.shape != M_arr.shape:
        raise ValueError("`H` and `M` must have the same shape.")
    if len(H_arr) < min_points:
        raise ValueError(f"Need at least {min_points} points; got {len(H_arr)}.")
    if method not in {"maxdev", "rms"}:
        raise ValueError("`method` must be either 'maxdev' or 'rms'.")

    # 3) Check monotonicity and reverse if strictly decreasing
    _check_monotonicity(H_arr)
    increasing = np.all(np.diff(H_arr) >= 0)
    decreasing = np.all(np.diff(H_arr) <= 0)

    if decreasing and not increasing:
        H_proc = H_arr[::-1].copy()
        M_proc = M_arr[::-1].copy()
        reversed_flag = True
    else:
        H_proc = H_arr
        M_proc = M_arr
        reversed_flag = False

    # 4) Find index of H closest to zero in the processed array
    start_idx = int(np.argmin(np.abs(H_proc)))

    # 5) Grow the window
    last_valid = start_idx
    n_total = len(H_proc)

    for end in range(start_idx + min_points - 1, n_total):
        H_seg = H_proc[start_idx : end + 1]
        M_seg = M_proc[start_idx : end + 1]

        # Fit line: M ≈ m * H + b
        m_try, b_try = np.polyfit(H_seg, M_seg, 1)
        residuals = M_seg - (m_try * H_seg + b_try)

        if method == "maxdev":
            # Check if every point is within margin
            if np.max(np.abs(residuals)) <= margin_val:
                last_valid = end
            else:
                break

        else:  # method == "rms"
            # Compute RMS error
            rmse = np.sqrt(np.mean(residuals**2))
            if rmse <= margin_val:
                last_valid = end
            else:
                break

    # 6) If no extension beyond start_idx, fail
    if last_valid == start_idx:
        raise ValueError(f"No linear segment found with method='{method}'.")

    # 7) Final fit on the maximal valid window
    H_final = H_proc[start_idx : last_valid + 1]
    M_final = M_proc[start_idx : last_valid + 1]
    m_opt, b_opt = np.polyfit(H_final, M_final, 1)

    # 8) Map the final index back if reversed
    if reversed_flag:
        orig_idx = len(H_proc) - 1 - last_valid
        Hmax_val = H_arr[orig_idx]
    else:
        Hmax_val = H_arr[last_valid]

    # 9) Return a single LinearSegmentProperties
    return LinearSegmentProperties(
        Mr=me.Mr(b_opt),
        Hmax=me.H(Hmax_val),
        gradient=m_opt * u.dimensionless_unscaled,
        _H=me.H(H, unit="A/m"),
        _M=me.Ms(M, unit="A/m"),
    )
