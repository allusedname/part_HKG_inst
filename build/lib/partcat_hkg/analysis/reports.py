from __future__ import annotations

import pandas as pd


def branch_accuracy_table(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"branch": k, "value": v} for k, v in metrics.items()])
