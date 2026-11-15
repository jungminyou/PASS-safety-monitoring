import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
if not os.environ.get("PASS_PRIVATE_DATA") or not os.environ.get("PASS_WORK_DIR"):
    raise RuntimeError(
        "Set PASS_PRIVATE_DATA and PASS_WORK_DIR to private directories outside the public package."
    )
PRIVATE_DATA = Path(os.environ["PASS_PRIVATE_DATA"]).resolve()
WORK = Path(os.environ["PASS_WORK_DIR"]).resolve()
SYNTHETIC = PACKAGE / "synthetic_data"


def _inside_package(folder):
    return folder == PACKAGE or PACKAGE in folder.parents


# The bundled synthetic dataset is the only input allowed inside the package.
if PRIVATE_DATA != SYNTHETIC and _inside_package(PRIVATE_DATA):
    raise ValueError("Restricted inputs must be outside the public package.")
if _inside_package(WORK):
    raise ValueError("Analysis outputs must be outside the public package.")
RESULTS, STATE, FIGURES = (WORK / name for name in ("results", "state", "figures"))
for folder in (RESULTS, STATE, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)
