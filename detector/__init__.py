# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.47"

from detector.data.explorer.explorer import Explorer
from detector.models import YOLO, YOLOWorld
from detector.utils import ASSETS, SETTINGS
from detector.utils.checks import check_yolo as checks
from detector.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "checks",
    "download",
    "settings",
    "Explorer",
)
