#!/bin/sh
set -eu
# See scripts/modules/rockem_bridge.py: mpiEminvTE2d is the EXPLICIT TE2D
# inversion engine. The ADI TE2D engine (mpiEminvADITE2d) fails the suite's
# own layered-model Green's-function validation (HZ ~178% error) where the
# explicit engine passes (~1-4%) - see rockem-suite's doc/examples/
# validate_layered_1d_model/README.md.
ROCKEM_SUITE_ROOT="${ROCKEM_SUITE_ROOT:-$HOME/software/new_rockem/rockem-suite}"
mpirun "$ROCKEM_SUITE_ROOT/bin/mpiEminvTE2d" inv.cfg
