import pkg_resources

from .bemcs import (
    plot_fields_no_elements,
    plot_fields,
    plot_element_geometry,
    standardize_elements,
    discretized_line,
)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
