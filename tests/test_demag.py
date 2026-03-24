"""Tests for demagnetization factor calculation."""

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest

from mammos_analysis.demag import demag_cuboid


def test_demag_cuboid_entities():
    """Test demag_cuboid with entities."""
    a = me.Entity("Length", 2.1, "mm")
    b = me.Entity("Length", 0.0022, "m")
    c = me.Entity("Length", 2.2e3, "um")
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(demag_cuboid(a, b, c), me.Entity)
    assert np.all(np.isclose(demag_cuboid(a, b, c).value, expected))


def test_demag_cuboid_quantities():
    """Test demag_cuboid with quantities."""
    a = u.Quantity(2.1, "mm")
    b = u.Quantity(0.0022, "m")
    c = u.Quantity(2.2e3, "um")
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(demag_cuboid(a, b, c), me.Entity)
    assert np.all(np.isclose(demag_cuboid(a, b, c).value, expected))


def test_demag_cuboid_float():
    """Test demag_cuboid with floating point numbers."""
    a = 2.1
    b = 2.2
    c = 2.2
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(demag_cuboid(a, b, c), me.Entity)
    assert np.all(np.isclose(demag_cuboid(a, b, c).value, expected))


def test_demag_cuboid_entity_quantity():
    """Test demag_cuboid with mix of entities and quantities."""
    a = me.Entity("Length", 2.1, "mm")
    b = u.Quantity(0.0022, "m")
    c = me.Entity("Length", 2.2e3, "um")
    expected = np.array([0.34374604, 0.32812698, 0.32812698])
    assert isinstance(demag_cuboid(a, b, c), me.Entity)
    assert np.all(np.isclose(demag_cuboid(a, b, c).value, expected))


def test_demag_cuboid_array():
    """Test demag_cuboid with objects containing array of elements."""
    a = me.Entity("Length", [2.1] * 3, "mm")
    b = u.Quantity([0.0022] * 3, "m")
    c = me.Entity("Length", [2.2e3] * 3, "um")
    expected = np.array([[0.34374604] * 3, [0.32812698] * 3, [0.32812698] * 3])
    assert isinstance(demag_cuboid(a, b, c), me.Entity)
    assert np.all(np.isclose(demag_cuboid(a, b, c).value, expected))


def test_demag_cuboid_sum():
    """Test for Dx + Dy + Dz = 1 (Eq. 2 in reference)."""
    dim = np.random.random(3)
    result = np.sum(demag_cuboid(*dim).value)
    assert np.isclose(result, 1)


def test_demag_cuboid_ValueError_units():
    """Test whether expected exceptions correctly occur."""
    a = me.Entity("Length", 2.1, "mm")
    b = u.Quantity(0.0022, "m")
    c = 2.2
    with pytest.raises(ValueError):
        demag_cuboid(a, b, c)


def test_demag_cuboid_ValueError_complex():
    """Test whether expected exceptions correctly occur."""
    a = 2.1
    b = 2.1 + 0j
    c = 2.2
    with pytest.raises(ValueError):
        demag_cuboid(a, b, c)


def test_demag_cuboid_ValueError_negative():
    """Test whether expected exceptions correctly occur."""
    a = 2.1
    b = -2.1
    c = 2.2
    with pytest.raises(ValueError):
        demag_cuboid(a, b, c)
