"""darm-vol: parsimonious finite-memory ARCH models (Connell 2026)."""
from .model import DARM
from .forecasting import Forecast
from .simulation import Simulation
from .results import DARMResults
from .kernels import kernel_weights

__version__ = "0.1.0"
__all__ = ["DARM", "DARMResults", "Forecast", "Simulation", "kernel_weights", "__version__"]
