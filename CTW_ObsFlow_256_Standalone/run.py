"""Run the packaged CTW observation-conditioned Flow from any directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    arguments = sys.argv[1:]
    if not any(value == "--config" or value.startswith("--config=") for value in arguments):
        arguments = ["--config", "config.json", *arguments]
    custom_data = any(value == "--data" or value.startswith("--data=") for value in arguments)
    fixed_initialization = any(
        value == "--initial_factors" or value.startswith("--initial_factors=")
        for value in arguments
    )
    if custom_data and not fixed_initialization:
        # The bundled initial factors belong only to example/cave.mat.
        arguments.extend(["--initial_factors", ""])
    sys.argv = [str(root / "run_ca_dps.py"), *arguments]
    from run_ca_dps import main as run_fusion

    run_fusion()


if __name__ == "__main__":
    main()
