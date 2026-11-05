"""Reconstruct source-verified Pantavou questionnaires and optional wearable features.

Third-party raw spreadsheets are not redistributed with this repository.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = ["air_temperature_c", "relative_humidity_pct", "wind_speed_m_s", "clothing_clo"]
RAD = ["shortwave_down_w_m2", "globe_temperature_c"]
ACTIVITY = ["mets_preceding"]
LABELS = np.arange(-3, 4)


def load_pantavou(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    raw.columns = [" ".join(str(col).split()) for col in raw.columns]
    raw["source_row_id"] = np.arange(len(raw)) + 2
    time_col = next(c for c in raw if c.startswith("Time ("))
    raw["timestamp_local"] = pd.to_datetime(raw["Date"]).dt.normalize() + pd.to_timedelta(raw[time_col].astype(str))
    raw = raw.rename(columns={
        "Participant": "user_id",
        "Thermal sensation vote": "tsv",
        "Tair (oC)": "air_temperature_c",
        "RH (%)": "relative_humidity_pct",
        "WSaver (m/s)": "wind_speed_m_s",
        "Iclo (clo)": "clothing_clo",
        "SR↓ (W/m2)": "shortwave_down_w_m2",
        "Tglobe (oC)": "globe_temperature_c",
    })
    raw["user_id"] = raw.user_id.astype(int)
    raw["tsv"] = pd.to_numeric(raw.tsv, errors="raise").astype(int)
    assert raw.tsv.isin(LABELS).all(), "TSV is not in [-3,+3]"
    # If the same person + minute was logged more than once, exclude the entire group.
    repeats = raw.duplicated(["user_id", "timestamp_local"], keep=False)
    clean = raw.loc[~repeats].sort_values(["user_id", "timestamp_local", "source_row_id"]).copy()
    clean["local_date"] = clean.timestamp_local.dt.strftime("%Y-%m-%d")
    assert len(raw) == 391 and len(clean) == 383 and clean.user_id.nunique() == 5
    return clean.reset_index(drop=True)


def add_wearables(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """METS measured no later than survey, max five minutes earlier.

    Diagnostic ONLY: contemporaneous wearables are NOT a pre-departure input.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    xl = pd.ExcelFile(path)
    pieces = []
    for uid in sorted(df.user_id.unique()):
        a = pd.read_excel(xl, sheet_name=f"Participant {uid}")
        time_col = next(c for c in a if c.startswith("Time ("))
        a["timestamp_local"] = pd.to_datetime(a["Date"]).dt.normalize() + pd.to_timedelta(a[time_col].astype(str))
        a["user_id"] = uid
        pieces.append(a[["timestamp_local", "user_id", "METS"]].rename(columns={"METS":"mets_preceding"}))
    sensors = pd.concat(pieces).sort_values("timestamp_local")
    left = df.sort_values("timestamp_local")
    result = pd.merge_asof(left, sensors, on="timestamp_local", by="user_id", direction="backward", tolerance=pd.Timedelta("5min"))
    assert len(result)==len(df) and result.source_row_id.is_unique
    return result.sort_values(["user_id", "timestamp_local", "source_row_id"]).reset_index(drop=True)


def subject_splits(df: pd.DataFrame):
    """Other persons train general model; test only days 2+ of held-out person."""
    for uid in sorted(df.user_id.unique()):
        train = df.loc[df.user_id != uid].copy()
        own = df.loc[df.user_id == uid].sort_values(["timestamp_local", "source_row_id"]).copy()
        first_day = sorted(own.local_date.unique())[0]
        test = own.loc[own.local_date != first_day].copy()
        assert len(train)>0 and len(test)>0 and test.timestamp_local.min()>own.timestamp_local.min()
        yield int(uid), train, own, test
