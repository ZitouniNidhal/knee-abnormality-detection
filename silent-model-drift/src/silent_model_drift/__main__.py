"""Module entry point for ``python -m silent_model_drift``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
