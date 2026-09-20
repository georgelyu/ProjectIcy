"""Package entry point for coupled falling-ice melting."""

from .examples.coupled_falling_ice_melting_2d import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
