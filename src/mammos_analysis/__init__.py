"""Post-processing (hysteresis loop, kuzmin, ...)."""

import importlib.metadata

from mammos_analysis import hysteresis as hysteresis
from mammos_analysis.kuzmin import kuzmin_properties as kuzmin_properties

__version__ = importlib.metadata.version(__package__)
