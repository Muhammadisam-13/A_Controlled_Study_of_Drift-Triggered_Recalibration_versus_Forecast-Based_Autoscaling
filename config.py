"""
Central path configuration for the whole pipeline.

Every path below is derived from this project's own location, so cloning
the repo anywhere "just works" for data/results/figures. The one exception
is KISIM_ROOT: the location of the separately-cloned KIS-S/KISim repo
(see README), which is genuinely machine-specific. Change it here (or set
the KISIM_ROOT environment variable) and every script picks it up.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- The only thing you should need to change per machine ---
# Defaults to a sibling "KISim" directory next to this repo, matching the
# README's `git clone https://github.com/GuilinDev/KISim ../KISim`.
KISIM_ROOT = Path(os.environ.get("KISIM_ROOT", PROJECT_ROOT.parent / "KISim"))

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
KISIM_RESULTS = KISIM_ROOT / "results" / "cpu_baseline" / "dynamic"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
