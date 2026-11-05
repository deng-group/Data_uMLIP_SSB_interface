"""
leaderboard of all benchmarked models

reads metrics.interface from each models/<family>/<model_key>.yml, sorts by eps_interface,
prints the table and saves it to benchmark/leaderboard.csv. models not run yet are skipped.

    python benchmark/leaderboard.py
"""
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML

HERE = Path(__file__).resolve().parent

rows = []
for yml in sorted((HERE / "models").glob("*/*.yml")):
    meta = YAML().load(yml)
    m = (meta.get("metrics") or {}).get("interface")
    if not m:
        continue
    row = {"model": meta.get("model_name", meta.get("model_key"))}
    row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
    row["excluded"] = len(m.get("excluded") or [])
    rows.append(row)

if not rows:
    raise SystemExit("No model has been benchmarked yet.")
table = pd.DataFrame(rows).set_index("model").sort_values("eps_interface")
table.to_csv(HERE / "leaderboard.csv")
print(table.to_string(float_format=lambda v: f"{v:.3f}"))
