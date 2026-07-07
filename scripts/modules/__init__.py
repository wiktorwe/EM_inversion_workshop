"""Modular script entry points used by the workshop GUI.

Submodules are imported explicitly by notebooks (e.g.
``from scripts.modules.workshop_config import load_config``). Avoid eager
imports here so Step 00 can load configuration before rockem-suite is
resolved.
"""

__all__ = [
    "analytic_1d_forward",
    "fd",
    "fd_visualization",
    "inversion",
    "rockem_bridge",
    "segy",
    "source",
    "survey",
    "workshop_config",
]
