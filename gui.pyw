"""Double-click entry point when .pyw uses the project's Python environment."""
from almanac.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["gui"]))
