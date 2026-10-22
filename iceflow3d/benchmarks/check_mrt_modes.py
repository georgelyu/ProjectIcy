"""Compatibility entry point for the current HOME CM-MRT viscosity check.

The former full-population eigenmode experiment is archived in git revision
7889ec4fea33cefca4404559e81a5d54f30b1d63. This runs the actual HOME kernels.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d.benchmarks.validate_home import main  # noqa: E402

if __name__ == "__main__":
    sys.argv.extend(["--mode", "viscosity"])
    main()
