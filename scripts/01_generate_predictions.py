import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "pipeline" / "01_generate_predictions.py"),
        run_name="__main__",
    )
