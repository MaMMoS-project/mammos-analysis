"""Postprocessing functions."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import mammos_entity
import mammos_entity as me
import mammos_units as u
import numbers
import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from scipy import optimize
import warnings


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class KuzminResult:
    """Result of Kuz'min magnetic properties estimation.

    Args:
        Ms: Temperature dependent Spontaneous Magnetisation (in A/m). Temperature is
            expressed in K.
        A: Temperature dependent Exchange Stiffness Constant (in J/m). Temperature is
            expressed in K.
        K1: Temperature dependent Uniaxial magnetocrystaling anisotropy (in A/m).
            Temperature is expressed in K.

    """

    Ms: Callable[[numbers.Real | u.Quantity], me.Entity]
    A: Callable[[numbers.Real | u.Quantity], me.Entity]
    K1: Callable[[numbers.Real | u.Quantity], me.Entity]
    Tc: me.Entity
    s: float


def kuzmin_properties(
    Ms: mammos_entity.Entity,
    T: mammos_entity.Entity,
    K1_0: mammos_entity.Entity,
) -> KuzminResult:
    """Evaluate micromagnetic intrinsic properties.

    If temperature T is given, evaluate them at that temperature.
    Otherwise, the three outputs `Ms`, `A`, and `K1` are going to be
    functions of temperature.

    Args:
        Ms: Spontaneous magnetisation data points.
        T: Temperature data points.
        K1_0: Magnetocrystalline anisotropy at temperature 0K.

    Returns:
        Intrinsic micromagnetic properties as functions of temperature.

    Raises:
        ValueError: Wrong unit.

    """
    if not isinstance(K1_0, u.Quantity) or K1_0.unit != u.J / u.m**3:
        K1_0 = me.Ku(K1_0, unit=u.J / u.m**3)

    # TODO: fix logic - assumption is that Ms is given at T=0K
    Ms_0 = me.Ms(Ms.value[0], unit=u.A / u.m)
    M_kuzmin = partial(kuzmin_formula, Ms_0.value)

    def residuals(params_, T_, M_):
        T_c_, s_ = params_
        return M_ - M_kuzmin(T_c_, s_, T_)

    with warnings.catch_warnings(action="ignore"):
        results = optimize.least_squares(
            residuals,
            (400, 0.5),
            args=(T.value, Ms.value),
            bounds=((0, 0), (np.inf, np.inf)),
            jac="3-point",
        )
    T_c, s = results.x
    T_c = T_c * u.K
    D = (
        0.1509
        * ((6 * u.constants.muB) / (s * Ms_0)) ** (2.0 / 3)
        * u.constants.k_B
        * T_c
    ).si
    A_0 = me.A(Ms_0 * D / (4 * u.constants.muB), unit=u.J / u.m)

    return KuzminResult(
        _Ms_function_of_temperature(Ms_0.value, T_c.value, s),
        _A_function_of_temperature(A_0, Ms_0.value, T_c.value, s),
        _K1_function_of_temperature(K1_0, Ms_0.value, T_c.value, s),
        me.Entity("ThermodynamicTemperature", value=T_c, unit="K"),
        s,
    )


def kuzmin_formula(Ms_0, T_c, s, T):
    """General Kuz'min formula.

    TODO: citation

    Args:
        Ms_0: Spontaneous magnetisation at zero Kelvin temperature.
        T_c: Curie temperature.
        s: Factor appearing in Kuz'min formula, to be optimised based on data.
        T: Temperature at which the magnetisation is evaluated.

    Returns:
        Spontaneous magnetisation at temperature T.

    """
    base = 1 - s * (T / T_c) ** 1.5 - (1 - s) * (T / T_c) ** 2.5
    out = np.zeros_like(T)
    # only compute base**(1/3) where T < T_c; elsewhere leave as zero
    np.power(base, 1 / 3, out=out, where=(T < T_c))
    return Ms_0 * out


class _A_function_of_temperature:
    def __init__(self, A_0, Ms_0, T_c, s):
        self.A_0 = A_0
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s

    def __repr__(self):
        return "A(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.A(
            self.A_0 * (kuzmin_formula(self.Ms_0, self.T_c, self.s, T) / self.Ms_0) ** 2
        )


class _K1_function_of_temperature:
    def __init__(self, K1_0, Ms_0, T_c, s):
        self.K1_0 = K1_0
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s

    def __repr__(self):
        return "K1(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.Ku(
            self.K1_0
            * (kuzmin_formula(self.Ms_0, self.T_c, self.s, T) / self.Ms_0) ** 3
        )


class _Ms_function_of_temperature:
    def __init__(self, Ms_0, T_c, s):
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s

    def __repr__(self):
        return "Ms(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.Ms(kuzmin_formula(self.Ms_0, self.T_c, self.s, T))
