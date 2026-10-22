"""Short alias: python -m iceflow3d.examples.dambreak_four_ice."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow3d.examples.dambreak_four_ice_melting_3d import main

if __name__ == "__main__":
    main()
