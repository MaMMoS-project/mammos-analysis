"""Postprocessing functions for micromagnetic property estimation."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import mammos_entity
import mammos_entity as me
import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from scipy import optimize

if TYPE_CHECKING:
    import astropy.units
    import mammos_entity
    import matplotlib
    import numpy


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class KuzminResult:
    """Result of Kuz'min magnetic properties estimation."""

    Ms: Callable[[numbers.Real | u.Quantity], me.Entity]
    """Callable returning temperature-dependent spontaneous magnetization."""
    A: Callable[[numbers.Real | u.Quantity], me.Entity]
    """Callable returning temperature-dependent exchange stiffness."""
    Tc: me.Entity
    """Curie temperature."""
    s: u.Quantity
    """Kuzmin parameter."""
    K1: Callable[[numbers.Real | u.Quantity], me.Entity] | None = None
    """Callable returning temperature-dependent uniaxial anisotropy."""

    def plot(
        self,
        T: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray | None = None,
    ) -> matplotlib.axes.Axes:
        """Create a plot for Ms, A, and K1 as a function of temperature."""
        ncols = 2 if self.K1 is None else 3
        w, h = figaspect(1 / ncols)
        default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(w, h))
        self.Ms.plot(T, ax[0], color=default_color_cycle[0])
        self.A.plot(T, ax[1], color=default_color_cycle[1])
        if self.K1 is not None:
            self.K1.plot(T, ax[2], color=default_color_cycle[2])
        return ax


def kuzmin_properties(
    Ms: mammos_entity.Entity,
    T: mammos_entity.Entity,
    Tc: mammos_entity.Entity | None = None,
    Ms_0: mammos_entity.Entity | None = None,
    K1_0: mammos_entity.Entity | None = None,
) -> KuzminResult:
    """Evaluate intrinsic micromagnetic properties using Kuz'min model.

    Computes Ms, A, and K1 as function of temperature by fitting the Kuzmin equation
    to Ms vs T. The attributes Ms, A and K1 in the returned object can be called to get
    values at arbitrary temperatures.

    K1 is only available in the output data if a zero-temperature value has been passed.
    If Ms_0 is None, the first value in the Ms series is taken as the
    zero temperature magnetization Ms_0.
    If Tc is None, it will be treated as an optimization variable
    and estimated during the fitting process.

    Args:
        Ms: Spontaneous magnetization data points as a me.Entity.
        T: Temperature data points as a me.Entity.
        K1_0: Magnetocrystalline anisotropy at 0 K as a me.Entity.
        TC: Curie-Temperature value as a me.Entity.
        Ms_0: Spontaneous magnetization at T=0 value as a me.Entity.

    Returns:
        KuzminResult with temperature-dependent Ms, A, K1 (optional),
        Curie temperature (optional), and exponent.

    Raises:
        ValueError: If K1_0 has incorrect unit.
    """
    if K1_0 is not None and (
        not isinstance(K1_0, me.Entity) or K1_0.unit != u.J / u.m**3
    ):
        K1_0 = me.Ku(K1_0, unit=u.J / u.m**3)

    if Ms_0 is None:
        Ms_0 = me.Ms(Ms.value[0], unit=u.A / u.m)

    # determine if Tc is optimized
    optimize_Tc = Tc is None

    # set init guess and bounds
    init_guess = [300, 0.5] if optimize_Tc else [0.5]
    bounds = ([0, 0], [np.inf, np.inf]) if optimize_Tc else ([0], [np.inf])

    def residuals(params, T_, M_):
        if optimize_Tc:
            Tc_, s_ = params
        else:
            Tc_ = Tc.value.item() if Tc.value.ndim == 0 else Tc.value[0][0]
            s_ = params[0]
        return M_ - kuzmin_formula(Ms_0.value, Tc_, s_, T_)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = optimize.least_squares(
            residuals,
            init_guess,
            args=(T.value, Ms.value),
            bounds=bounds,
            jac="3-point",
        )

    if optimize_Tc:
        T_c, s = results.x

    else:
        T_c_val = Tc.value.item() if Tc.value.ndim == 0 else Tc.value[0][0]
        T_c = T_c_val
        s = results.x[0]

    T_c = T_c * u.K

    D = (
        0.1509
        * ((6 * u.constants.muB) / (s * Ms_0.q)) ** (2.0 / 3)
        * u.constants.k_B
        * T_c
    ).si
    A_0 = me.A(Ms_0 * D / (4 * u.constants.muB), unit=u.J / u.m)

    if K1_0 is not None:
        K1 = _K1_function_of_temperature(K1_0, Ms_0.value, T_c.value, s, T)
    else:
        K1 = None

    return KuzminResult(
        Ms=_Ms_function_of_temperature(Ms_0.value, T_c.value, s, T),
        A=_A_function_of_temperature(A_0, Ms_0.value, T_c.value, s, T),
        K1=K1,
        Tc=me.Tc(value=T_c, unit="K"),
        s=s * u.dimensionless_unscaled,
    )


def kuzmin_formula(Ms_0, T_c, s, T):
    """Compute spontaneous magnetization at temperature T using Kuz'min formula.

    Kuz’min, M.D., Skokov, K.P., Diop, L.B. et al. Exchange stiffness of ferromagnets.
    Eur. Phys. J. Plus 135, 301 (2020). https://doi.org/10.1140/epjp/s13360-020-00294-y

    Args:
        Ms_0: Spontaneous magnetization at 0 K.
        T_c: Curie temperature.
        s: Kuzmin exponent parameter.
        T: Temperature(s) for evaluation.

    Returns:
        Spontaneous magnetization at temperature T as an array.
    """
    if isinstance(Ms_0, me.Entity):
        Ms_0 = Ms_0.value
    if isinstance(T_c, me.Entity):
        T_c = T_c.value
    if isinstance(T, me.Entity):
        T = T.value
    base = 1 - s * (T / T_c) ** 1.5 - (1 - s) * (T / T_c) ** 2.5
    out = np.zeros_like(T, dtype=np.float64)
    # only compute base**(1/3) where T < T_c; elsewhere leave as zero
    np.power(base, 1 / 3, out=out, where=(T_c > T))
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

    def __init__(self, A_0, Ms_0, T_c, s, T):
        self.A_0 = A_0
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s
        self._T = T

    def __repr__(self):
        return "A(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.A(
            self.A_0.q
            * (kuzmin_formula(self.Ms_0, self.T_c, self.s, T) / self.Ms_0) ** 2
        )

    def plot(
        self,
        T: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot A as a function of temperature using Kuzmin formula."""
        if not ax:
            _, ax = plt.subplots()
        if T is None:
            T = np.linspace(min(self._T.value), max(self._T.value), 100)
        if not isinstance(T, me.Entity):
            T = me.T(T)
        A = self(T)
        ax.plot(T.q, A.q, **kwargs)
        ax.set_xlabel(T.axis_label)
        ax.set_ylabel(A.axis_label)
        ax.grid()
        return ax


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

    def __init__(self, K1_0, Ms_0, T_c, s, T):
        self.K1_0 = K1_0
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s
        self._T = T

    def __repr__(self):
        return "K1(T)"

    def __call__(self, T: numbers.Real | u.Quantity) -> me.Entity:
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.Ku(
            self.K1_0.q
            * (kuzmin_formula(self.Ms_0, self.T_c, self.s, T) / self.Ms_0) ** 3
        )

    def plot(
        self,
        T: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot K1 as a function of temperature using Kuzmin formula."""
        if not ax:
            _, ax = plt.subplots()
        if T is None:
            T = np.linspace(min(self._T.value), max(self._T.value), 100)
        if not isinstance(T, me.Entity):
            T = me.T(T)
        K1 = self(T)
        ax.plot(T.q, K1.q, **kwargs)
        ax.set_xlabel(T.axis_label)
        ax.set_ylabel(K1.axis_label)
        ax.grid()
        return ax


class _Ms_function_of_temperature:
    """Callable for temperature-dependent spontaneous magnetization Ms(T).

    Attributes:
        Ms_0: Spontaneous magnetization at 0 K.
        T_c: Curie temperature.
        s: Kuzmin exponent parameter.

    Call:
        Returns Ms(T) as a me.Entity for given temperature T.
    """

    def __init__(
        self,
        Ms_0: mammos_entity.Entity,
        T_c: mammos_entity.Entity,
        s: astropy.units.Quantity,
        T: mammos_entity.Entity,
    ):
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s
        self._T = T

    def __repr__(self):
        return "Ms(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.Ms(kuzmin_formula(self.Ms_0, self.T_c, self.s, T))

    def plot(
        self,
        T: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot Ms as a function of temperature using Kuzmin formula."""
        if not ax:
            _, ax = plt.subplots()
        if T is None:
            T = np.linspace(min(self._T.value), max(self._T.value), 100)
        if not isinstance(T, me.Entity):
            T = me.T(T)
        Ms = self(T)
        ax.plot(T.q, Ms.q, **kwargs)
        ax.set_xlabel(T.axis_label)
        ax.set_ylabel(Ms.axis_label)
        ax.grid()
        return ax
