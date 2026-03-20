try:
    from mhrqi._version import __version__
except ImportError:
    __version__ = "0.1.0"

from mhrqi.core.representation import MHRQI
from mhrqi.core.results import MHRQIResult

__all__ = ["MHRQI", "MHRQIResult", "__version__"]
