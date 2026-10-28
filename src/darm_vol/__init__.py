"""darm-vol: parsimonious finite-memory ARCH models (Connell 2026)."""
from .forecasting import Forecast
from .kernels import kernel_weights
from .model import DARM
from .results import DARMResults
from .simulation import Simulation

__version__ = "0.1.0"
__all__ = ["DARM", "DARMResults", "Forecast", "Simulation", "kernel_weights", "__version__"]
