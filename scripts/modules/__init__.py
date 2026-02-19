"""Modular script entry points used by the workshop GUI."""

from . import fd
from . import fd_visualization
from . import inversion
from . import segy
from . import source
from . import survey

__all__ = ["fd", "fd_visualization", "inversion", "segy", "source", "survey"]
