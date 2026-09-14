#!/usr/bin/env python3
"""
CLI entry point for Phase 4 — Step 6: Exploratory Data Analysis (EDA).
Executes end-to-end dataset analysis from PostgreSQL, exporting figures to data/analysis/figures/,
summary tables to data/analysis/tables/, and compiling the markdown report to data/analysis/reports/.

Usage:
    python scripts/run_eda.py
    python scripts/run_eda.py --category Cars
    python scripts/run_eda.py --ml-eligible-only
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eda.report import main

if __name__ == "__main__":
    main()
