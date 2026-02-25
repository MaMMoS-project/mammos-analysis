"""Functions calculating demagnetization factors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import astropy.units
    import mammos_entity
    import numpy

import numpy as np
from mammos_entity import Entity


def demag_cuboid(
    x1: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray,
    x2: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray,
    x3: mammos_entity.Entity | astropy.units.Quantity | numpy.ndarray,
) -> mammos_entity.Entity:
    """Calculate demagnetization factors of a rectangular cuboid.

    Equation 1 from A. Aharoni, J. Appl. Phys. 83, 3422 (1998).
    https://doi.org/10.1063/1.367113

    Args:
        x1: Full side length of rectangular cuboid in direction 1
        x2: Full side length of rectangular cuboid in direction 2
        x3: Full side length of rectangular cuboid in direction 3

    Returns:
        Demagnetizing factors along each dimension. Order of dimensions is the
        same as for input arguments.

    Raises:
        ValueError: If arguments with and without unit are mixed.
        ValueError: If arguments are complex
        ValueError: If arguments are negative
    """
    # convert all dimensions to same unit and extract as values
    if all([hasattr(i, "unit") for i in [x1, x2, x3]]):
        ref_unit = x1.unit
        x1, x2, x3 = [
            i.q.to(ref_unit).value if isinstance(i, Entity) else i.to(ref_unit).value
            for i in [x1, x2, x3]
        ]
    # don't allow ambiguous situation when only some arguments have unit
    elif any([hasattr(i, "unit") for i in [x1, x2, x3]]):
        hint = {True: "contains a unit", False: "does not contain a unit"}
        raise ValueError(
            f"""Only some arguments contain a unit while others do not. This is
            ambiguous and thus not supported.
            x1 is of type {type(x1)} and {hint[hasattr(x1, "unit")]}.
            x2 is of type {type(x2)} and {hint[hasattr(x2, "unit")]}.
            x3 is of type {type(x3)} and {hint[hasattr(x3, "unit")]}."""
        )

    # check for complex dimensions of cuboid
    if np.any(np.iscomplexobj([x1, x2, x3])):
        hint = ""
        for key, value in {"x1": x1, "x2": x2, "x3": x3}.items():
            if np.iscomplexobj(value):
                hint = hint + f"{key} appears to be a complex object.\n"
        raise ValueError(f"Complex cuboid dimensions are not allowed.\n{hint}")

    # check for negative dimensions of cuboid
    if np.any(np.array([x1, x2, x3]) < 0):
        hint = ""
        for key, value in {"x1": x1, "x2": x2, "x3": x3}.items():
            if value < 0:
                hint = hint + f"{key} appears to be a negative number.\n"
        raise ValueError(f"Negative cuboid dimensions are not allowed.\n{hint}")

    def _calc_D(x1, x2, x3):
        # the expression takes input as half of the semi-axes
        a = 0.5 * x1
        b = 0.5 * x2
        c = 0.5 * x3
        # define some convenience terms
        a2 = a * a
        b2 = b * b
        c2 = c * c
        abc = a * b * c
        ab = a * b
        ac = a * c
        bc = b * c
        r_abc = np.sqrt(a2 + b2 + c2)
        r_ab = np.sqrt(a2 + b2)
        r_bc = np.sqrt(b2 + c2)
        r_ac = np.sqrt(a2 + c2)
        # compute the factor
        pi_Dz = (
            ((b2 - c2) / (2 * bc)) * np.log((r_abc - a) / (r_abc + a))
            + ((a2 - c2) / (2 * ac)) * np.log((r_abc - b) / (r_abc + b))
            + (b / (2 * c)) * np.log((r_ab + a) / (r_ab - a))
            + (a / (2 * c)) * np.log((r_ab + b) / (r_ab - b))
            + (c / (2 * a)) * np.log((r_bc - b) / (r_bc + b))
            + (c / (2 * b)) * np.log((r_ac - a) / (r_ac + a))
            + 2 * np.arctan2(ab, c * r_abc)
            + (a2 * a + b2 * b - 2 * c2 * c) / (3 * abc)
            + ((a2 + b2 - 2 * c2) / (3 * abc)) * r_abc
            + (c / ab) * (r_ac + r_bc)
            - (r_ab * r_ab * r_ab + r_bc * r_bc * r_bc + r_ac * r_ac * r_ac) / (3 * abc)
        )
        # divide out the factor of pi
        D = pi_Dz / np.pi
        return D

    D = (_calc_D(x2, x3, x1), _calc_D(x1, x3, x2), _calc_D(x1, x2, x3))

    return Entity("DemagnetizingFactor", D)
