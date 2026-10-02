"""Shared config for the SOP evaluation (rolling-origin backtest).

Origins are the LAST OBSERVED time_idx at forecast time. Each origin forecasts
time_idx origin+1 .. origin+30. The four 30-day test windows are disjoint and
tile the last 120 days of the data (2024-09-03 .. 2024-12-31).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "supply_chain_data.csv")
RESULTS = os.path.join(ROOT, "eval_sop", "results")
HORIZON = 30
MAX_T = 1460  # last time_idx in the data (1461 days, 2021-01-01..2024-12-31)
ORIGINS = [MAX_T - HORIZON * k for k in (4, 3, 2, 1)]  # [1340, 1370, 1400, 1430]
QUANTILES = [0.1, 0.5, 0.9]
SEEDS = [0, 1, 2]
