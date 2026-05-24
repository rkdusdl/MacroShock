from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# 0. Local paths
# ============================================================
# Edit this path when running locally on Windows.
RAW_DIR = Path(r"C:\Users\gayeon\Dropbox\가연\2026\Macro\0_DB\shock\monetary_raw")
OUT_DIR = RAW_DIR / "processed_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If True, no-event months inside each event-study source's own sample are coded as zero.
# Months outside the source's sample remain missing after outer merge.
FILL_NO_EVENT_ZERO = True


# ============================================================
# 1. Date / aggregation utilities
# ============================================================
def month_start(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def quarter_start(s):
    return pd.to_datetime(s).dt.to_period("Q").dt.start_time


def year_start(s):
    return pd.to_datetime(s).dt.to_period("Y").dt.start_time


def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in ["year", "quarter", "month"]:
        if c in out.columns:
            out = out.drop(columns=[c])
    cols = ["date"] + [c for c in out.columns if c != "date"]
    out = out[cols]
    out.insert(1, "year", out["date"].dt.year)
    out.insert(2, "quarter", out["date"].dt.quarter)
    out.insert(3, "month", out["date"].dt.month)
    return out


def outer_merge_on_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = None
    for f in frames:
        if f is None or f.empty:
            continue
        f = f.copy()
        f["date"] = pd.to_datetime(f["date"])
        out = f if out is None else out.merge(f, on="date", how="outer")
    if out is None:
        return pd.DataFrame(columns=["date"])
    return out.sort_values("date").reset_index(drop=True)


def to_monthly_event(
    df: pd.DataFrame,
    date_col: str,
    prefix: str,
    cols: Optional[list[str]] = None,
    count_col: Optional[str] = None,
    fill_no_event_zero: bool = FILL_NO_EVENT_ZERO,
) -> pd.DataFrame:
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.loc[x[date_col].notna()].copy()
    x["date"] = month_start(x[date_col])

    if cols is None:
        cols = [
            c
            for c in x.columns
            if c not in [date_col, "date"] and pd.api.types.is_numeric_dtype(x[c])
        ]
    cols = [c for c in cols if c in x.columns]
    if not cols:
        return pd.DataFrame(columns=["date"])

    g = x.groupby("date")[cols].sum(min_count=1)
    g.columns = [f"{prefix}{c}" for c in g.columns]

    if count_col:
        g[f"{prefix}{count_col}"] = x.groupby("date").size()

    if len(g) == 0:
        return g.reset_index()

    grid = pd.DataFrame({"date": pd.date_range(g.index.min(), g.index.max(), freq="MS")}).set_index("date")
    out = grid.join(g, how="left")

    if fill_no_event_zero:
        event_count_name = f"{prefix}{count_col}" if count_col else None
        if event_count_name and event_count_name in out.columns:
            no_event = out[event_count_name].isna()
        else:
            no_event = out.isna().all(axis=1)
        out.loc[no_event, :] = out.loc[no_event, :].fillna(0)

    return out.reset_index()


def aggregate_period(monthly: pd.DataFrame, freq: str) -> pd.DataFrame:
    x = monthly.copy()
    if freq == "Q":
        x["date"] = quarter_start(x["date"])
    elif freq == "Y":
        x["date"] = year_start(x["date"])
    else:
        raise ValueError(freq)

    num_cols = [
        c
        for c in x.columns
        if c not in ["date", "year", "quarter", "month"] and pd.api.types.is_numeric_dtype(x[c])
    ]
    out = x.groupby("date", as_index=False)[num_cols].sum(min_count=1)
    return add_time_cols(out)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


# ============================================================
# 2. Generic readers
# ============================================================
def read_csv_flexible(obj, **kwargs) -> pd.DataFrame:
    return normalize_column_names(pd.read_csv(obj, **kwargs))


def read_excel_flexible(obj, **kwargs) -> pd.DataFrame:
    return normalize_column_names(pd.read_excel(obj, **kwargs))


def read_stata_flexible(obj, **kwargs) -> pd.DataFrame:
    return normalize_column_names(pd.read_stata(obj, **kwargs))


def read_table_by_suffix(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return read_csv_flexible(path)
        if suffix in [".xlsx", ".xls"]:
            # First sheet by default; RR stable files are usually simple tables.
            return read_excel_flexible(path)
        if suffix == ".dta":
            return read_stata_flexible(path)
    except Exception:
        return None
    return None


def read_table_from_zip_member(z: zipfile.ZipFile, member: str) -> Optional[pd.DataFrame]:
    suffix = Path(member).suffix.lower()
    try:
        raw = z.read(member)
        bio = BytesIO(raw)
        if suffix == ".csv":
            return read_csv_flexible(bio)
        if suffix in [".xlsx", ".xls"]:
            return read_excel_flexible(bio)
        if suffix == ".dta":
            return read_stata_flexible(bio)
    except Exception:
        return None
    return None


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["date", "Date", "DATE", "time", "Time", "Datetime", "datetime", "fomc", "FOMC"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column that parses reasonably as dates
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() >= max(5, int(0.5 * len(df))):
                return c
        except Exception:
            pass
    return None


def infer_rr_frequency(name: str, df: pd.DataFrame, date_col: str) -> str:
    low = name.lower()
    if any(k in low for k in ["month", "monthly", "m.", "_m", "-m"]):
        return "M"
    if any(k in low for k in ["quarter", "quarterly", "q.", "_q", "-q"]):
        return "Q"
    if any(k in low for k in ["annual", "year", "yearly", "a.", "_a", "-a"]):
        return "Y"

    d = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    if len(d) < 3:
        return "M"
    # If dates are mostly quarter starts or separated by 70-100 days, call quarterly.
    gap_days = d.diff().dt.days.dropna().median()
    if 70 <= gap_days <= 110:
        return "Q"
    if 300 <= gap_days <= 400:
        return "Y"
    return "M"


# ============================================================
# 3. Updated Romer-Romer loader
# ============================================================
def standardize_rr_table(df: pd.DataFrame, source_name: str) -> Optional[tuple[str, pd.DataFrame]]:
    """Return (freq, standardized RR dataframe) or None.

    Handles two common cases:
    1. Stable deposit style: date, resid, resid_romer, resid_full.
    2. GitHub style: fomc, rr_original, rr_update, DFFR.
    """
    x = normalize_column_names(df)
    lower_map = {c.lower(): c for c in x.columns}

    rr_like_cols = []
    for c in x.columns:
        cl = c.lower()
        if cl in ["resid", "resid_romer", "resid_full", "rr_original", "rr_update", "dffr"]:
            rr_like_cols.append(c)
    if not rr_like_cols:
        return None

    date_col = find_date_column(x)
    if date_col is None:
        return None

    x = x.copy()
    x["date"] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.loc[x["date"].notna()].copy()
    if x.empty:
        return None

    rename = {}
    for c in x.columns:
        cl = c.lower()
        if cl == "resid":
            rename[c] = "rr_resid"
        elif cl == "resid_romer":
            rename[c] = "rr_resid_romer"
        elif cl == "resid_full":
            rename[c] = "rr_resid_full"
        elif cl == "rr_original":
            rename[c] = "rr_original"
        elif cl == "rr_update":
            rename[c] = "rr_update"
        elif cl == "dffr":
            rename[c] = "rr_DFFR"

    keep = ["date"] + list(rename.keys())
    x = x[keep].rename(columns=rename)

    for c in x.columns:
        if c != "date":
            x[c] = pd.to_numeric(x[c], errors="coerce")

    freq = infer_rr_frequency(source_name, x, "date")

    # Stable monthly/quarterly/yearly files are already at their target frequency.
    if freq == "M":
        x["date"] = month_start(x["date"])
        x = x.groupby("date", as_index=False).sum(min_count=1)
    elif freq == "Q":
        x["date"] = quarter_start(x["date"])
        x = x.groupby("date", as_index=False).sum(min_count=1)
    elif freq == "Y":
        x["date"] = year_start(x["date"])
        x = x.groupby("date", as_index=False).sum(min_count=1)

    return freq, x.sort_values("date").reset_index(drop=True)


def load_updated_romer_romer(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Search RAW_DIR recursively for updated Romer-Romer files.

    Priority:
    - stable deposit files with resid/resid_romer/resid_full
    - GitHub rrshocks.csv with rr_original/rr_update/DFFR

    Returns a dictionary with possible keys: M, Q, Y.
    """
    found: dict[str, list[tuple[str, pd.DataFrame]]] = {"M": [], "Q": [], "Y": []}

    # Regular files
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".csv", ".xlsx", ".xls", ".dta", ".zip"]:
            continue
        if path.name.startswith("~$"):
            continue

        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as z:
                    for member in z.namelist():
                        if Path(member).suffix.lower() not in [".csv", ".xlsx", ".xls", ".dta"]:
                            continue
                        # Skip large unrelated replication files unless the member looks RR-like.
                        ml = member.lower()
                        if not any(k in ml for k in ["romer", "rr", "shock", "resid"]):
                            continue
                        df = read_table_from_zip_member(z, member)
                        if df is None:
                            continue
                        out = standardize_rr_table(df, f"{path.name}/{member}")
                        if out is not None:
                            freq, rr = out
                            found[freq].append((f"{path.name}/{member}", rr))
            except Exception:
                pass
            continue

        df = read_table_by_suffix(path)
        if df is None:
            continue
        out = standardize_rr_table(df, path.name)
        if out is not None:
            freq, rr = out
            found[freq].append((path.name, rr))

    # Choose best candidate for each frequency.
    # Score stable deposit columns above GitHub columns; newer/larger table second.
    result: dict[str, pd.DataFrame] = {}
    for freq, candidates in found.items():
        if not candidates:
            continue

        def score(item):
            name, df = item
            cols = set(df.columns)
            stable_score = int("rr_resid_full" in cols) * 100 + int("rr_resid_romer" in cols) * 50 + int("rr_resid" in cols) * 25
            github_score = int("rr_update" in cols) * 10 + int("rr_original" in cols) * 5
            n_score = min(len(df), 10000) / 10000
            return stable_score + github_score + n_score

        best_name, best_df = sorted(candidates, key=score, reverse=True)[0]
        result[freq] = best_df
        print(f"Loaded updated RR {freq} from: {best_name} | shape={best_df.shape}")

    # If only monthly RR found, aggregate to Q/Y for convenience.
    if "M" in result and "Q" not in result:
        result["Q"] = aggregate_period(add_time_cols(result["M"]), "Q").drop(columns=["year", "quarter", "month"])
    if "M" in result and "Y" not in result:
        result["Y"] = aggregate_period(add_time_cols(result["M"]), "Y").drop(columns=["year", "quarter", "month"])

    return result


def load_legacy_romer_romer(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Fallback: old Romer & Romer zip used in the previous script."""
    p = raw_dir / "Romer and Romer Data and Programs.zip"
    if not p.exists():
        return None
    try:
        with zipfile.ZipFile(p) as z:
            xnames = [n for n in z.namelist() if n.endswith("Romer&Romer_Data.xlsx")]
            if not xnames:
                return None
            with z.open(xnames[0]) as f:
                rr = pd.read_excel(f, sheet_name="Monthly Data", header=24)
        rr = rr.rename(columns={rr.columns[0]: "date"})
        rr["date"] = pd.to_datetime(rr["date"], errors="coerce")
        keep = [c for c in ["date", "RRDUMMY", "RRDUMMYORIG"] if c in rr.columns]
        rr = rr[keep].dropna(subset=["date"]).copy()
        rr = rr.rename(columns={"RRDUMMY": "rr_legacy_RRDUMMY", "RRDUMMYORIG": "rr_legacy_RRDUMMYORIG"})
        rr["date"] = month_start(rr["date"])
        return rr.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print("Legacy RR load failed:", e)
        return None


# ============================================================
# 4. Main loaders for each shock family
# ============================================================
monthly_frames: list[pd.DataFrame] = []
sources: list[list] = []

# 4.1 Jarocinski-Karadi Fed MP/CBI, monthly already.
p = RAW_DIR / "shocks_fed_jk_m.csv"
if p.exists():
    jk = pd.read_csv(p)
    jk["date"] = pd.to_datetime(dict(year=jk["year"], month=jk["month"], day=1))
    jk = jk.drop(columns=["year", "month"])
    jk = jk.rename(columns={c: f"jk_{c}" for c in jk.columns if c != "date"})
    monthly_frames.append(jk)
    sources.append(["jk", p.name, "Fed JK MP/CBI monthly shock", str(jk.date.min().date()), str(jk.date.max().date()), len(jk)])

# 4.2 Identkurto non-Gaussian Fed four shocks, bp and standardized event files.
for filename, prefix, label in [
    ("U1bp.csv", "ident_bp_", "Identkurto Fed four shocks, bp-normalized"),
    ("U1s.csv", "ident_s_", "Identkurto Fed four shocks, one-s.d. standardized"),
]:
    p = RAW_DIR / filename
    if p.exists():
        u = pd.read_csv(p)
        f = to_monthly_event(u, "Time", prefix, cols=["u1", "u2", "u3", "u4"], count_col="event_count")
        monthly_frames.append(f)
        sources.append([prefix.rstrip("_"), p.name, label, str(pd.to_datetime(u.Time).min().date()), str(pd.to_datetime(u.Time).max().date()), len(u)])

# 4.3 Monetary policy surprises from USMPD / Acosta-style factors.
p = RAW_DIR / "monetary-policy-surprises.zip"
if p.exists():
    with zipfile.ZipFile(p) as z:
        mps = pd.read_csv(z.open("mps.csv"), parse_dates=["Date"])
        mins = pd.read_csv(z.open("mps_minutes.csv"), parse_dates=["Date"])
    f = to_monthly_event(mps, "Date", "mps_", cols=["STMT", "PC", "ME"], count_col="event_count")
    monthly_frames.append(f)
    sources.append(["mps", "monetary-policy-surprises.zip/mps.csv", "USMPD-derived statement/press-conference/monetary-event surprises", str(mps.Date.min().date()), str(mps.Date.max().date()), len(mps)])
    f = to_monthly_event(mins, "Date", "mps_minutes_", cols=["MIN"], count_col="event_count")
    monthly_frames.append(f)
    sources.append(["mps_minutes", "monetary-policy-surprises.zip/mps_minutes.csv", "USMPD-derived FOMC minutes surprise", str(mins.Date.min().date()), str(mins.Date.max().date()), len(mins)])



# 4.3b Bauer-Swanson / monetary-policy-surprises-data.xlsx, update 2023.
#      This workbook has FOMC event-level and monthly monetary policy surprise data.
#      We include the update-2023 sheets only to avoid duplicate original-vintage columns.
def sanitize_colname(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"[^0-9A-Za-z]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def parse_excel_or_date_series(s: pd.Series) -> pd.Series:
    """Parse either Excel serial dates or regular date strings."""
    x = s.copy()
    numeric = pd.to_numeric(x, errors="coerce")
    # Excel serial dates are typically around 30000-50000 for modern dates.
    if numeric.notna().sum() >= max(3, int(0.5 * len(x))) and numeric.dropna().between(20000, 60000).mean() > 0.8:
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(x, errors="coerce")


def numeric_columns_after_cleaning(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Convert non-excluded columns to numeric where possible and return numeric columns."""
    for c in df.columns:
        if c in exclude:
            continue
        df[c] = pd.to_numeric(df[c].replace({"NA": np.nan, "NaN": np.nan, "nan": np.nan, "": np.nan}), errors="coerce")
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


p = RAW_DIR / "monetary-policy-surprises-data.xlsx"
if p.exists():
    # Monthly update sheet: already monthly. Prefix bs_m_ to distinguish from USMPD mps_*.
    try:
        bs_m = pd.read_excel(p, sheet_name="Monthly (update 2023)")
        bs_m = normalize_column_names(bs_m).dropna(axis=1, how="all")
        if {"Year", "Month"}.issubset(bs_m.columns):
            bs_m["date"] = pd.to_datetime(
                dict(
                    year=pd.to_numeric(bs_m["Year"], errors="coerce"),
                    month=pd.to_numeric(bs_m["Month"], errors="coerce"),
                    day=1,
                ),
                errors="coerce",
            )
            bs_m = bs_m.loc[bs_m["date"].notna()].copy()
            cols = numeric_columns_after_cleaning(bs_m, exclude=["Year", "Month", "date"])
            keep = ["date"] + cols
            bs_m_out = bs_m[keep].copy()
            bs_m_out = bs_m_out.rename(columns={c: f"bs_m_{sanitize_colname(c)}" for c in cols})
            bs_m_out = bs_m_out.groupby("date", as_index=False).sum(min_count=1)
            monthly_frames.append(bs_m_out)
            sources.append([
                "bs_mps_monthly_update2023",
                p.name + "/Monthly (update 2023)",
                "Bauer-Swanson-style monthly monetary policy surprises and macro controls, update 2023",
                str(bs_m_out.date.min().date()),
                str(bs_m_out.date.max().date()),
                len(bs_m_out),
            ])
    except Exception as e:
        print("Failed to load Monthly (update 2023) from monetary-policy-surprises-data.xlsx:", e)

    # FOMC update sheet: event-level. Aggregate by monthly sum. Prefix bs_fomc_.
    try:
        bs_f = pd.read_excel(p, sheet_name="FOMC (update 2023)")
        bs_f = normalize_column_names(bs_f).dropna(axis=1, how="all")
        if "Date" in bs_f.columns:
            bs_f["date"] = parse_excel_or_date_series(bs_f["Date"])
            bs_f = bs_f.loc[bs_f["date"].notna()].copy()
            exclude = ["Date", "date", "Time"]
            cols = numeric_columns_after_cleaning(bs_f, exclude=exclude)
            f = to_monthly_event(bs_f, "date", "bs_fomc_", cols=cols, count_col="event_count")
            monthly_frames.append(f)
            sources.append([
                "bs_mps_fomc_update2023",
                p.name + "/FOMC (update 2023)",
                "Bauer-Swanson-style FOMC event-level monetary policy surprises, update 2023; monthly sums",
                str(bs_f.date.min().date()),
                str(bs_f.date.max().date()),
                len(bs_f),
            ])
    except Exception as e:
        print("Failed to load FOMC (update 2023) from monetary-policy-surprises-data.xlsx:", e)


# 4.4 USMPD key raw high-frequency surprises, monthly sums.
p = RAW_DIR / "USMPD.xlsx"
if p.exists():
    me = pd.read_excel(p, sheet_name="Monetary Events")
    key_cols = [
        c for c in [
            "MP1", "MP2", "FF1", "FF2", "OIS1Y", "OIS2Y",
            "UST2Y", "UST5Y", "UST10Y", "SP500", "SPFUT", "DXY", "SEP", "Unscheduled", "PC"
        ] if c in me.columns
    ]
    f = to_monthly_event(me, "date_time", "usmpd_me_", cols=key_cols, count_col="event_count")
    monthly_frames.append(f)
    sources.append(["usmpd_me", p.name + "/Monetary Events", "USMPD raw key asset-price surprises around monetary events", str(me.date_time.min().date()), str(me.date_time.max().date()), len(me)])

# 4.5 Updated Romer-Romer. This is now preferred over the legacy RR zip.
rr_by_freq = load_updated_romer_romer(RAW_DIR)
if "M" in rr_by_freq:
    monthly_frames.append(rr_by_freq["M"])
    rr_m = rr_by_freq["M"]
    sources.append(["rr_updated", "auto-detected updated RR", "Updated Romer-Romer monetary shocks; baseline column usually rr_resid_full", str(rr_m.date.min().date()), str(rr_m.date.max().date()), len(rr_m)])
else:
    rr_legacy = load_legacy_romer_romer(RAW_DIR)
    if rr_legacy is not None:
        monthly_frames.append(rr_legacy)
        sources.append(["rr_legacy", "Romer and Romer Data and Programs.zip/Romer&Romer_Data.xlsx", "Legacy Romer & Romer narrative monetary shock series", str(rr_legacy.date.min().date()), str(rr_legacy.date.max().date()), len(rr_legacy)])

# 4.6 UKMPESD factor shocks, monthly sums, clearly prefixed as UK.
p = RAW_DIR / "measuring-monetary-policy-in-the-uk-the-ukmpesd.xlsx"
if p.exists():
    uk = pd.read_excel(p, sheet_name="factors")
    f = to_monthly_event(uk, "Datetime", "uk_", cols=["Target", "Path", "QE"], count_col="event_count")
    monthly_frames.append(f)
    sources.append(["ukmpesd", p.name + "/factors", "UK MP event-study factor shocks: Target, Path, QE", str(uk.Datetime.min().date()), str(uk.Datetime.max().date()), len(uk)])


# ============================================================
# 5. Build final monthly / quarterly / yearly datasets
# ============================================================
monthly = outer_merge_on_date(monthly_frames)
monthly = add_time_cols(monthly)

quarterly = aggregate_period(monthly, "Q")
yearly = aggregate_period(monthly, "Y")

# If updated RR provides frequency-specific Q/Y files, overwrite aggregated RR columns with official Q/Y values.
def replace_rr_columns(base: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    rr_cols = [c for c in out.columns if c.startswith("rr_")]
    out = out.drop(columns=rr_cols, errors="ignore")
    off = official.copy()
    off["date"] = pd.to_datetime(off["date"])
    out = out.merge(off, on="date", how="outer")
    return add_time_cols(out.sort_values("date").reset_index(drop=True))

if "Q" in rr_by_freq:
    q_official = rr_by_freq["Q"].copy()
    q_official["date"] = quarter_start(q_official["date"])
    quarterly = replace_rr_columns(quarterly, q_official)

if "Y" in rr_by_freq:
    y_official = rr_by_freq["Y"].copy()
    y_official["date"] = year_start(y_official["date"])
    yearly = replace_rr_columns(yearly, y_official)


# ============================================================
# 6. Metadata / QA
# ============================================================
def classify_family(c: str) -> str:
    if c.startswith("jk_"):
        return "JK Fed MP/CBI"
    if c.startswith("ident_bp_"):
        return "Identkurto bp"
    if c.startswith("ident_s_"):
        return "Identkurto standardized"
    if c.startswith("mps_minutes_"):
        return "USMPD MPS minutes"
    if c.startswith("mps_"):
        return "USMPD MPS"
    if c.startswith("usmpd_me_"):
        return "USMPD raw monetary events"
    if c.startswith("rr_"):
        return "Updated Romer-Romer narrative"
    if c.startswith("uk_"):
        return "UKMPESD"
    return "other"


var_rows = []
for c in monthly.columns:
    if c in ["date", "year", "quarter", "month"]:
        continue
    var_rows.append([c, classify_family(c), "Monthly value; quarterly/yearly files aggregate by sum except updated RR Q/Y official files if detected."])

dictionary = pd.DataFrame(var_rows, columns=["variable", "family", "aggregation_note"])
sources_df = pd.DataFrame(sources, columns=["source_id", "file", "description", "source_start", "source_end", "source_rows"])


def qa_rows(df: pd.DataFrame, freq: str) -> list[list]:
    rows = []
    for c in df.columns:
        if c in ["date", "year", "quarter", "month"]:
            continue
        s = df[c]
        rows.append([
            freq,
            c,
            int(s.notna().sum()),
            float(s.mean(skipna=True)) if s.notna().any() else np.nan,
            float(s.std(skipna=True)) if s.notna().sum() > 1 else np.nan,
            float(s.min(skipna=True)) if s.notna().any() else np.nan,
            float(s.max(skipna=True)) if s.notna().any() else np.nan,
        ])
    return rows


qa = pd.DataFrame(
    qa_rows(monthly, "monthly") + qa_rows(quarterly, "quarterly") + qa_rows(yearly, "yearly"),
    columns=["frequency", "variable", "nonmissing", "mean", "sd", "min", "max"],
)


# ============================================================
# 7. Write outputs
# ============================================================
monthly.to_csv(OUT_DIR / "monetary_shocks_monthly.csv", index=False)
quarterly.to_csv(OUT_DIR / "monetary_shocks_quarterly.csv", index=False)
yearly.to_csv(OUT_DIR / "monetary_shocks_yearly.csv", index=False)
dictionary.to_csv(OUT_DIR / "monetary_shocks_dictionary.csv", index=False)
sources_df.to_csv(OUT_DIR / "monetary_shocks_sources.csv", index=False)
qa.to_csv(OUT_DIR / "monetary_shocks_qa_summary.csv", index=False)

# Excel workbook for quick inspection.
with pd.ExcelWriter(OUT_DIR / "monetary_shocks_final.xlsx", engine="openpyxl") as writer:
    monthly.to_excel(writer, sheet_name="monthly", index=False)
    quarterly.to_excel(writer, sheet_name="quarterly", index=False)
    yearly.to_excel(writer, sheet_name="yearly", index=False)
    dictionary.to_excel(writer, sheet_name="dictionary", index=False)
    sources_df.to_excel(writer, sheet_name="sources", index=False)
    qa.to_excel(writer, sheet_name="qa_summary", index=False)

readme = f"""# Final monetary shock dataset

Generated by `process_monetary_shocks_v3_with_bs_mps_xlsx.py`.

Core aggregation rule:

    shock_period = sum(shock_event_or_month within period)

Exception:

    If updated Romer-Romer official quarterly / yearly files are detected,
    those official frequency-specific series replace mechanically aggregated RR columns.

No-event months inside each event-study source's own sample are coded as zero.
Months outside each source's sample remain missing.

Main outputs:
- monetary_shocks_monthly.csv: {monthly.shape[0]} rows x {monthly.shape[1]} columns
- monetary_shocks_quarterly.csv: {quarterly.shape[0]} rows x {quarterly.shape[1]} columns
- monetary_shocks_yearly.csv: {yearly.shape[0]} rows x {yearly.shape[1]} columns
- monetary_shocks_final.xlsx
- monetary_shocks_dictionary.csv
- monetary_shocks_sources.csv
- monetary_shocks_qa_summary.csv

Recommended baseline variables:
- JK high-frequency MP/CBI split: jk_MP_median, jk_CBI_median
- Identkurto four-shock non-Gaussian identification: ident_s_u1, ident_s_u2, ident_s_u3, ident_s_u4
- USMPD-derived event surprise: mps_ME, plus mps_STMT and mps_PC
- Bauer-Swanson update-2023 surprise workbook: bs_m_MPS, bs_m_MPS_ORTH, bs_fomc_MPS, bs_fomc_MPS_ORTH
- Updated Romer-Romer narrative shock: rr_resid_full if available; fallback rr_update or legacy RR columns otherwise
- UK shocks: uk_Target, uk_Path, uk_QE
"""
(OUT_DIR / "README_monetary_shocks.md").write_text(readme, encoding="utf-8")

print("Wrote outputs to", OUT_DIR)
print("monthly", monthly.shape, monthly.date.min().date(), monthly.date.max().date())
print("quarterly", quarterly.shape, quarterly.date.min().date(), quarterly.date.max().date())
print("yearly", yearly.shape, yearly.date.min().date(), yearly.date.max().date())
print("RR columns monthly:", [c for c in monthly.columns if c.startswith("rr_")])

print("BS/MPS xlsx columns monthly:", [c for c in monthly.columns if c.startswith("bs_")][:30], "... total", len([c for c in monthly.columns if c.startswith("bs_")]))
