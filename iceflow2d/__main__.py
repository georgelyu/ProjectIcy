"""Package entry point for the rigid-ice dam-break example."""

from .examples.dam_break_2d import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
