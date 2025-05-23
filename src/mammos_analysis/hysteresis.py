"""Hysteresis analysis and postprocessing functions."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mammos_units
    import mammos_entity

import numbers
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
    h = _check_unit(H, u.A / u.m, equivalencies=u.magnetic_flux_field()).value
    m = _check_unit(M, u.A / u.m, equivalencies=u.magnetic_flux_field()).value

    sign_changes_m = np.where(np.diff(np.sign(m)))[0]
    sign_changes_h = np.where(np.diff(np.sign(h)))[0]

    if len(sign_changes_m) == 0:
        raise ValueError("Failed to calculate Hc.")

    if len(sign_changes_h) == 0:
        raise ValueError("Failed to calculate Mr.")

    index_before = sign_changes_m[0]
    index_after = sign_changes_m[0] + 1
    Hc = -1 * np.interp(
        0,
        [m[index_before], m[index_after]],
        [h[index_before], h[index_after]],
    )

    index_before = sign_changes_h[0]
    index_after = sign_changes_h[0] + 1
    Mr = np.interp(
        0,
        [h[index_before], h[index_after]],
        [m[index_before], m[index_after]],
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
    H = _check_unit(H, u.T, equivalencies=u.magnetic_flux_field())
    M = _check_unit(M, u.T, equivalencies=u.magnetic_flux_field())
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


def _check_unit(
    x: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
    unit: mammos_units.Unit,
    equivalencies: u.Equivalency | None = None,
) -> mammos_units.Quantity:
    """Check unit of a certain object.

    If the object `x` is a mammos_entity.Entity, the ontology label will be lost.

    Args:
        x: Object whose unit is to be checked.
        unit: Desired unit
        equivalencies: Astropy equivalencies to be used.

    Returns:
        Object with the right unit.

    """
    if not isinstance(x, u.Quantity) or x.unit != unit:
        x = x.to(unit, equivalencies=equivalencies)
    return x
