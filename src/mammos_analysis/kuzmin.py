"""Postprocessing functions for micromagnetic property estimation."""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import TYPE_CHECKING
from warnings import warn

import mammos_entity
import mammos_entity as me
import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    import astropy.units
    import mammos_entity
    import mammos_units
    import matplotlib
    import numpy


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class KuzminResult:
    """Result of Kuz'min magnetic properties estimation."""

    Ms: Callable[[numbers.Real | u.Quantity], me.Entity]
    """Callable returning temperature-dependent :entity:`SpontaneousMagnetization`."""
    A: Callable[[numbers.Real | u.Quantity], me.Entity]
    """Callable returning temperature-dependent :entity:`ExchangeStiffnessConstant`."""
    Tc: me.Entity
    """:entity:`CurieTemperature`."""
    s: u.Quantity
    """Kuzmin parameter."""
    K1: Callable[[numbers.Real | u.Quantity], me.Entity] | None = None
    """Callable returning temperature-dependent uniaxial anisotropy."""

    def plot(
        self,
        T: mammos_entity.Entity
        | mammos_units.Quantity
        | numpy.typing.ArrayLike
        | None = None,
        celsius: bool = False,
    ) -> matplotlib.axes.Axes:
        """Create a plot for Ms, A, and K1 as a function of temperature.

        Args:
            T: If specified, the entities are plotted against this array. Otherwise, a
                uniform array of 100 points is generated between the minimum and the
                maximum available data.
            celsius: If True, plots the temperature in degree Celsius.
        """
        ncols = 2 if self.K1 is None else 3
        w, h = figaspect(1 / ncols)
        default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(w, h))
        self.Ms.plot(T, ax[0], celsius=celsius, color=default_color_cycle[0])
        self.A.plot(T, ax[1], celsius=celsius, color=default_color_cycle[1])
        if self.K1 is not None:
            self.K1.plot(T, ax[2], celsius=celsius, color=default_color_cycle[2])
        return ax


def kuzmin_properties(
    Ms: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    T: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    Tc: mammos_entity.Entity | mammos_units.Quantity | numbers.Real | None = None,
    Ms_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real | None = None,
    K1_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real | None = None,
    Tc_initial_guess: mammos_entity.Entity
    | mammos_units.Quantity
    | numbers.Real
    | None = None,
    Ms_0_initial_guess: mammos_entity.Entity
    | mammos_units.Quantity
    | numbers.Real
    | None = None,
    s_initial_guess: numbers.Real = 0.5,
) -> KuzminResult:
    """Evaluate intrinsic micromagnetic properties using Kuz’min model.

    Computes Ms, A, and K1 as function of temperature by fitting the Kuz’min equation
    to Ms vs T. The attributes Ms, A and K1 in the returned object can be called to get
    values at arbitrary temperatures.

    K1 is only available in the output data if the value of the zero-temperature
    uniaxial anisotropy constant K1_0 has been passed.

    If Ms_0 is None, then we check if the first temperature value is zero. If so, we
    Ms_0 corresponding to the first value of Ms. Otherwise, its value is fitted from
    Kuz'min curve.

    If Tc is None, it will be treated as an optimization variable and estimated during
    the fitting process via curve fitting.

    If Ms_0 is fitted and `Ms_0_initial_guess` is not defined, we start the optimization
    from the value 1.2 * max(Ms_0).

    If Tc is fitted and `Tc_initial_guess` is not defined, we start the optimization
    from the highest temperature value T such that the corresponding Ms is
    higher than 0.1 * max(Ms).

    Args:
        Ms: :entity:`SpontaneousMagnetization`.
            If no unit is provided, values are interpreted as 'A / m'.
        T: :entity:`ThermodynamicTemperature`.
            If no unit is provided, values are interpreted as 'K'.
        K1_0: :entity:`MagnetocrystallineAnisotropyConstantK1` or
            :entity:`UniaxialAnisotropyConstant` at T = 0 K.
            If no unit is provided, values are interpreted as 'J / m^3'.
        Tc: :entity:`CurieTemperature`.
            If no unit is provided, values are interpreted as 'K'.
        Ms_0: :entity:`SpontaneousMagnetization` at T = 0 K.
            If no unit is provided, values are interpreted as 'A / m'.
        Tc_initial_guess: Initial guess for Tc
            :entity:`CurieTemperature` (if optimized).
            If no unit is provided, values are interpreted as 'K'.
        Ms_0_initial_guess: Initial guess for Ms_0
            :entity:`SpontaneousMagnetization` (if optimized).
            If no unit is provided, values are interpreted as 'A / m'.
        s_initial_guess: Initial guess for the parameter `s` appearing in the
            Kuz'min fit.

    Returns:
        KuzminResult with temperature-dependent Ms, A, K1 (optional),
        Curie temperature (optional), and exponent.

    Raises:
        ValueError: Value of Ms at zero temperature is not given.
        ValueError: If K1_0 has incorrect unit.
    """
    Ms = me._entity.from_compatible(
        "SpontaneousMagnetization", "A / m", Ms=Ms, enforce_unit=True
    )
    T = me._entity.from_compatible(
        "ThermodynamicTemperature", "K", T=T, enforce_unit=True
    )

    if K1_0 is not None:
        K1_0 = me._entity.from_compatible(
            "MagnetocrystallineAnisotropyConstantK1",
            "J / m^3",
            compatible_entities=("UniaxialAnisotropyConstant",),
            K1_0=K1_0,
            enforce_unit=True,
        )
    if Tc is not None:
        Tc = me._entity.from_compatible(
            "CurieTemperature", "K", Tc=Tc, enforce_unit=True
        )
    if Ms_0_initial_guess is not None:
        Ms_0_initial_guess = me._entity.from_compatible(
            "SpontaneousMagnetization",
            "A/m",
            Ms_0_initial_guess=Ms_0_initial_guess,
            enforce_unit=True,
        )
    if Tc_initial_guess is not None:
        Tc_initial_guess = me._entity.from_compatible(
            "CurieTemperature",
            "K",
            Tc_initial_guess=Tc_initial_guess,
            enforce_unit=True,
        )
    if Ms_0 is not None:
        Ms_0 = me._entity.from_compatible(
            "SpontaneousMagnetization", "A / m", Ms_0=Ms_0, enforce_unit=True
        )

    # We initialize initial guess and bounds for s.
    # If Ms_0 and Tc needs to be optimized, too,
    # we expand these two variables.
    initial_guess = [s_initial_guess]
    bounds = ([0], [np.inf])

    if Ms_0 is not None:
        optimize_Ms_0 = False
        if Ms_0_initial_guess is not None:
            warn(
                f"The user defined Ms_0_initial_guess={Ms_0_initial_guess} for the "
                f"optimizer even though the value Ms_0={Ms_0} was given.",
                stacklevel=2,
            )
    else:
        if np.isclose(T.value[0], 0):
            optimize_Ms_0 = False
            Ms_0 = me.Ms(Ms.value[0], unit=u.A / u.m)
        else:
            optimize_Ms_0 = True
            # We set the first value of data vector Ms
            # as initial guess and lower bound for Ms_0.
            if Ms_0_initial_guess is not None:
                initial_guess.append(Ms_0_initial_guess.value)
            else:
                # As default initial guess for Ms_0 we choose 1.2 * max(Ms)
                initial_guess.append(Ms.value.max() * 1.2)
            bounds[0].append(Ms.value.max() * 0.8)  # Ms_0 lower bound: 80% of max Ms
            bounds[1].append(np.inf)  # Ms_0 upper bound: inf

    if Tc is None:
        optimize_Tc = True
        if Tc_initial_guess is not None:
            initial_guess.append(Tc_initial_guess.value)
        else:
            # If Tc is not given, we set the initial guess as the
            # maximum temperature T such that Ms is 10% of its maximum value.
            initial_guess.append(T.value[Ms.q > 0.1 * Ms.q.max()].max())
        bounds[0].append(0)  # Tc lower bound: 0
        bounds[1].append(np.inf)  # Tc upper bound: inf
    else:
        optimize_Tc = False
        if Tc_initial_guess is not None:
            warn(
                f"The user defined Tc_initial_guess={Tc_initial_guess} for the "
                f"optimizer even though the value Tc={Tc} was given.",
                stacklevel=2,
            )
        Tc = Tc.value.flatten()[0] if Tc.value.ndim > 0 else Tc.value
        Tc = me.Entity("CurieTemperature", value=Tc)

    def F(T_, *params):
        s_ = params[0]
        Ms_0_ = params[1] if optimize_Ms_0 else Ms_0.value
        Tc_ = params[-1] if optimize_Tc else Tc.value
        return kuzmin_formula(Ms_0_, Tc_, s_, T_).value

    results = curve_fit(
        F, T.value, Ms.value, p0=initial_guess, bounds=bounds, jac="3-point"
    )

    p_opt = results[0]
    s = p_opt[0]
    if optimize_Ms_0:
        Ms_0 = me.Ms(p_opt[1])
    if optimize_Tc:
        Tc = me.Tc(p_opt[-1])

    D = (
        0.1509
        * ((6 * u.constants.muB) / (s * Ms_0.q)) ** (2.0 / 3)
        * u.constants.k_B
        * Tc.q
    ).si
    A_0 = me.A(Ms_0 * D / (4 * u.constants.muB), unit=u.J / u.m)

    if K1_0 is not None:
        K1 = _K1_function_of_temperature(K1_0, Ms_0.value, Tc.value, s, T)
    else:
        K1 = None

    return KuzminResult(
        Ms=_Ms_function_of_temperature(Ms_0.value, Tc.value, s, T),
        A=_A_function_of_temperature(A_0, Ms_0.value, Tc.value, s, T),
        K1=K1,
        Tc=Tc,
        s=s * u.dimensionless_unscaled,
    )


def kuzmin_formula(
    Ms_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
    T_c: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
    s: numbers.Real,
    T: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
) -> mammos_entity.Entity:
    r"""Compute spontaneous magnetization at temperature T using Kuz'min formula.

    The formula approximates the spontaneous magnetization :math:`M_s(T)` for
    :math:`0 < T < T_c` as

    .. math::

       M_s(T) = M_{s}(0) \left[ 1 - s \left( \frac{T}{T_c} \right)^{3/2}
       - (1-s) \left( \frac{T}{T_c} \right)^{5/2} \right]^{1/3}

    where :math:`M_{s}(0)` is the spontaneous magnetization at T = 0 K, :math:`T_c`
    is the Curie temperature, and :math:`s` is an adjustable parameter.

    Kuz’min, M.D., Skokov, K.P., Diop, L.B. et al. Exchange stiffness of ferromagnets.
    Eur. Phys. J. Plus 135, 301 (2020). https://doi.org/10.1140/epjp/s13360-020-00294-y

    Args:
        T: Temperature(s) for evaluation :entity:`ThermodynamicTemperature`.
            If no unit is provided, values are interpreted as 'K'.
        Ms_0: :entity:`SpontaneousMagnetization` at T = 0 K.
            If no unit is provided, values are interpreted as 'A / m'.
        T_c: :entity:`CurieTemperature`.
            If no unit is provided, values are interpreted as 'K'.
        s: Kuz’min exponent parameter.

    Returns:
        :entity:`SpontaneousMagnetization` at temperature(s) T.
    """
    Ms_0 = me._entity.from_compatible("SpontaneousMagnetization", "A / m", Ms_0=Ms_0)
    T_c = me._entity.from_compatible(
        "CurieTemperature", "K", T_c=T_c, enforce_unit=True
    )
    T = me._entity.from_compatible(
        "ThermodynamicTemperature", "K", T=T, enforce_unit=True
    )

    if not np.isscalar(Ms_0.value):
        raise ValueError("Argument Ms_0 must be a scalar spontaneous magnetization.")

    if not np.isscalar(T_c.value):
        raise ValueError("Argument T_c must be a scalar Curie temperature.")

    if not np.isscalar(s):
        raise ValueError("Argument s must be a scalar.")

    base = 1 - s * (T.q / T_c.q) ** 1.5 - (1 - s) * (T.q / T_c.q) ** 2.5

    out = np.zeros_like(base, dtype=np.float64)
    np.cbrt(base, out=out, where=T_c.q > T.q)  # compute cubic root of base

    return me.Ms((Ms_0.q * out).reshape(T.q.shape))


class _A_function_of_temperature:
    """Callable for temperature-dependent exchange stiffness A(T).

    Attributes:
        A_0: :entity:`ExchangeStiffnessConstant` at 0 K.
            If no unit is provided, values are interpreted as 'J/m'.
        Ms_0: :entity:`SpontaneousMagnetization` at 0 K.
            If no unit is provided, values are interpreted as 'A/m'.
        T_c: :entity:`CurieTemperature`.
            If no unit is provided, values are interpreted as 'K'.
        s: Kuzmin exponent parameter.

    Call:
        Returns A(T) as a me.Entity for given temperature T.
    """

    def __init__(
        self,
        A_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        Ms_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        T_c: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        s: numbers.Real,
        T: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    ):
        self.Ms_0 = me._entity.from_compatible(
            "SpontaneousMagnetization", "A / m", Ms_0=Ms_0
        )
        self.T_c = me._entity.from_compatible("CurieTemperature", "K", T_c=T_c)
        self._T = me._entity.from_compatible("ThermodynamicTemperature", "K", T=T)
        self.A_0 = me._entity.from_compatible(
            "ExchangeStiffnessConstant", "J/m", A_0=A_0
        )
        self.s = s

    def __repr__(self):
        return "A(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.A(
            self.A_0.q
            * (kuzmin_formula(self.Ms_0, self.T_c, self.s, T).q / self.Ms_0) ** 2,
            self.A_0.unit,
        )

    def plot(
        self,
        T: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        celsius: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot A as a function of temperature using Kuzmin formula.

        Args:
            T: If :entity:`ThermodynamicTemperature` is specified, the
                :entity:`ExchangeStiffnessConstant` is plotted against this array.
                Otherwise, a uniform array of 100 points is generated between the
                minimum and the maximum available data.
            ax: optional matplotlib ``Axes`` instance to plot on an existing subplot.
            celsius: If True, plots the temperature in degree Celsius.
            **kwargs: Additional plotting arguments.
        """
        if not ax:
            _, ax = plt.subplots()
        if T is None:
            T = np.linspace(min(self._T.value), max(self._T.value), 100)
        if not isinstance(T, me.Entity):
            T = me.T(T)
        A = self(T)
        if celsius:
            Tq = T.q.to("Celsius", equivalencies=u.temperature())
            T_label = T.axis_label.replace("(K)", "(°C)")
        else:
            Tq = T.q
            T_label = T.axis_label
        ax.plot(Tq, A.q, **kwargs)
        ax.set_xlabel(T_label)
        ax.set_ylabel(A.axis_label)
        ax.grid()
        return ax


class _K1_function_of_temperature:
    """Callable for temperature-dependent uniaxial anisotropy K1(T).

    Attributes:
        K1_0: :entity:`MagnetocrystallineAnisotropyConstantK1` at 0 K.
        Ms_0: :entity:`SpontaneousMagnetization` at 0 K.
        T_c: :entity:`CurieTemperature`.
        s: Kuzmin exponent parameter.

    Call:
        Returns :entity:`MagnetocrystallineAnisotropyConstantK1`
            for given temperature T.
    """

    def __init__(
        self,
        K1_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        Ms_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        T_c: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        s: numbers.Real,
        T: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    ):
        self.Ms_0 = me._entity.from_compatible(
            "SpontaneousMagnetization", "A / m", Ms_0=Ms_0
        )
        self.T_c = me._entity.from_compatible("CurieTemperature", "K", T_c=T_c)
        self._T = me._entity.from_compatible("ThermodynamicTemperature", "K", T=T)
        self.K1_0 = me._entity.from_compatible(
            "MagnetocrystallineAnisotropyConstantK1",
            "J/m^3",
            compatible_entities=("UniaxialAnisotropyConstant",),
            K1_0=K1_0,
        )
        self.s = s

    def __repr__(self):
        return "K1(T)"

    def __call__(self, T: numbers.Real | u.Quantity) -> me.Entity:
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.K1(
            self.K1_0.q
            * (kuzmin_formula(self.Ms_0, self.T_c, self.s, T).q / self.Ms_0) ** 3
        )

    def plot(
        self,
        T: mammos_entity.Entity | mammos_units.Quantity | numpy.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        celsius: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot K1 as a function of temperature using Kuzmin formula.

        Args:
            T: If specified, the uniaxial anisotropy is plotted against this array.
                Otherwise, a uniform array of 100 points is generated between the
                minimum and the maximum available data.
            ax: optional matplotlib ``Axes`` instance to plot on an existing subplot.
            celsius: If True, plots the temperature in degree Celsius.
            **kwargs: Additional plotting arguments.
        """
        if not ax:
            _, ax = plt.subplots()
        if T is None:
            T = np.linspace(min(self._T.value), max(self._T.value), 100)
        if not isinstance(T, me.Entity):
            T = me.T(T)
        K1 = self(T)
        if celsius:
            Tq = T.q.to("Celsius", equivalencies=u.temperature())
            T_label = T.axis_label.replace("(K)", "(°C)")
        else:
            Tq = T.q
            T_label = T.axis_label
        ax.plot(Tq, K1.q, **kwargs)
        ax.set_xlabel(T_label)
        ax.set_ylabel(K1.axis_label)
        ax.grid()
        return ax


class _Ms_function_of_temperature:
    """Callable for temperature-dependent spontaneous magnetization Ms(T).

    Attributes:
        Ms_0: :entity:`SpontaneousMagnetization` at 0 K.
        T_c: :entity:`CurieTemperature`.
        s: Kuzmin exponent parameter.

    Call:
        Returns Ms(T) as a me.Entity for given temperature T.
    """

    def __init__(
        self,
        Ms_0: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        T_c: mammos_entity.Entity | mammos_units.Quantity | numbers.Real,
        s: numbers.Real,
        T: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    ):

        self.Ms_0 = me._entity.from_compatible(
            "SpontaneousMagnetization", "A / m", Ms_0=Ms_0
        )
        self.T_c = me._entity.from_compatible("CurieTemperature", "K", T_c=T_c)
        self._T = me._entity.from_compatible("ThermodynamicTemperature", "K", T=T)
        self.s = s

    def __repr__(self):
        return "Ms(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.Ms(kuzmin_formula(self.Ms_0, self.T_c, self.s, T).q.to("kA/m"))

    def plot(
        self,
        T: mammos_entity.Entity | mammos_units.Quantity | numpy.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        celsius: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot Ms as a function of temperature using Kuzmin formula.

        Args:
            T: If specified, the spontaneous magnetization is plotted against this
                array. Otherwise, a uniform array of 100 points is generated between the
                minimum and the maximum available data.
            ax: optional matplotlib ``Axes`` instance to plot on an existing subplot.
            celsius: If True, plots the temperature in degree Celsius.
            **kwargs: Additional plotting arguments.
        """
        if not ax:
            _, ax = plt.subplots()
        if T is None:
            T = np.linspace(min(self._T.value), max(self._T.value), 100)
        if not isinstance(T, me.Entity):
            T = me.T(T)
        Ms = self(T)
        if celsius:
            Tq = T.q.to("Celsius", equivalencies=u.temperature())
            T_label = T.axis_label.replace("(K)", "(°C)")
        else:
            Tq = T.q
            T_label = T.axis_label
        ax.plot(Tq, Ms.q, **kwargs)
        ax.set_xlabel(T_label)
        ax.set_ylabel(Ms.axis_label)
        ax.grid()
        return ax
