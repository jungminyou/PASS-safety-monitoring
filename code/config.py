import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
if not os.environ.get("PASS_PRIVATE_DATA") or not os.environ.get("PASS_WORK_DIR"):
    raise RuntimeError(
        "Set PASS_PRIVATE_DATA and PASS_WORK_DIR to private directories outside the public package."
    )
PRIVATE_DATA = Path(os.environ["PASS_PRIVATE_DATA"]).resolve()
WORK = Path(os.environ["PASS_WORK_DIR"]).resolve()
for folder in (PRIVATE_DATA, WORK):
    if folder == PACKAGE or PACKAGE in folder.parents:
        raise ValueError(
            "Restricted inputs and analysis outputs must be outside the public package."
        )
RESULTS, STATE, FIGURES = (WORK / name for name in ("results", "state", "figures"))
for folder in (RESULTS, STATE, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)
