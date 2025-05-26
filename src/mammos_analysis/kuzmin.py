"""Postprocessing functions for micromagnetic property estimation."""

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

    Attributes:
        Ms: Callable returning temperature-dependent spontaneous magnetization.
        A: Callable returning temperature-dependent exchange stiffness.
        K1: Callable returning temperature-dependent uniaxial anisotropy.
        Tc: Curie temperature.
        s: Kuzmin parameter.
    """

    Ms: Callable[[numbers.Real | u.Quantity], me.Entity]
    A: Callable[[numbers.Real | u.Quantity], me.Entity]
    K1: Callable[[numbers.Real | u.Quantity], me.Entity]
    Tc: me.Entity
    s: u.Quantity


def kuzmin_properties(
    Ms: mammos_entity.Entity,
    T: mammos_entity.Entity,
    K1_0: mammos_entity.Entity,
) -> KuzminResult:
    """Evaluate intrinsic micromagnetic properties using Kuz'min model.

    If temperature data T is provided, the intrinsic properties are
    evaluated at those temperatures.
    Otherwise, Ms, A, and K1 are callables of temperature.

    Args:
        Ms: Spontaneous magnetization data points as a me.Entity.
        T: Temperature data points as a me.Entity.
        K1_0: Magnetocrystalline anisotropy at 0 K as a me.Entity.

    Returns:
        KuzminResult with temperature-dependent or evaluated values, Curie temperature,
        and exponent.

    Raises:
        ValueError: If K1_0 has incorrect unit.
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
        s * u.dimensionless_unscaled,
    )


def kuzmin_formula(Ms_0, T_c, s, T):
    """Compute spontaneous magnetization at temperature T using Kuz'min formula.

    TODO: add citation

    Args:
        Ms_0: Spontaneous magnetization at 0 K.
        T_c: Curie temperature.
        s: Kuzmin exponent parameter.
        T: Temperature(s) for evaluation.

    Returns:
        Spontaneous magnetization at temperature T as an array.
    """
    base = 1 - s * (T / T_c) ** 1.5 - (1 - s) * (T / T_c) ** 2.5
    out = np.zeros_like(T)
    # only compute base**(1/3) where T < T_c; elsewhere leave as zero
    np.power(base, 1 / 3, out=out, where=(T < T_c))
    return Ms_0 * out


class _A_function_of_temperature:
    """Callable for temperature-dependent exchange stiffness A(T).

    Attributes:
        A_0: Exchange stiffness at 0 K.
        Ms_0: Spontaneous magnetization at 0 K.
        T_c: Curie temperature.
        s: Kuzmin exponent parameter.

    Call:
        Returns A(T) as a me.Entity for given temperature T.
    """

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
    """Callable for temperature-dependent uniaxial anisotropy K1(T).

    Attributes:
        K1_0: Anisotropy constant at 0 K.
        Ms_0: Spontaneous magnetization at 0 K.
        T_c: Curie temperature.
        s: Kuzmin exponent parameter.

    Call:
        Returns K1(T) as a me.Entity for given temperature T.
    """

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
    """Callable for temperature-dependent spontaneous magnetization Ms(T).

    Attributes:
        Ms_0: Spontaneous magnetization at 0 K.
        T_c: Curie temperature.
        s: Kuzmin exponent parameter.

    Call:
        Returns Ms(T) as a me.Entity for given temperature T.
    """

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
