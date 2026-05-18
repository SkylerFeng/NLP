import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "pipeline" / "02_compute_similarity.py"),
        run_name="__main__",
    )
