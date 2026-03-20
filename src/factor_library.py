from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = Path.home() / "Downloads"

FACTOR_DATASETS = {
    "Japan 5 Factors": {
        "filename": "Japan_5_Factors.csv",
        "remote_zip_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_5_Factors_CSV.zip",
    },
    "North America 5 Factors": {
        "filename": "North_America_5_Factors.csv",
        "remote_zip_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/North_America_5_Factors_CSV.zip",
    },
}

PORTFOLIO_DATASETS = {
    "なし": None,
    "Japan 32 Portfolios": {
        "filename": "Japan_32_Portfolios_ME_INV(TA)_OP_2x4x4.csv",
        "section_hint": "Average Value Weighted Returns -- Monthly",
    },
    "10 Industry Portfolios": {
        "filename": "10_Industry_Portfolios_Daily.csv",
        "section_hint": "Average Value Weighted Returns -- Daily",
    },
}

FIVE_FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _candidate_paths(filename: str) -> list[Path]:
    return [
        REPO_ROOT / filename,
        REPO_ROOT / "data" / filename,
        DOWNLOADS_DIR / filename,
    ]


def _read_text_from_path(filename: str) -> str | None:
    for path in _candidate_paths(filename):
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return None


def _read_remote_zip_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = next((name for name in zf.namelist() if name.lower().endswith(".csv")), None)
        if not csv_name:
            raise ValueError("Could not find CSV file in remote zip archive.")
        return zf.read(csv_name).decode("utf-8", errors="ignore")


def _clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([-99.99, -999, -999.99], np.nan, inplace=True)
    return df


def _collect_numeric_rows(lines: list[str], start_idx: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines[start_idx:]:
        if not line.strip():
            if rows:
                break
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0].isdigit():
            if rows:
                break
            continue
        rows.append(parts)
    return rows


def _build_date_index(date_series: pd.Series) -> pd.Series:
    sample = str(date_series.iloc[0]).strip()
    if len(sample) == 6:
        return pd.to_datetime(date_series.astype(str), format="%Y%m")
    if len(sample) == 8:
        return pd.to_datetime(date_series.astype(str), format="%Y%m%d")
    raise ValueError(f"Unsupported date format: {sample}")


def parse_five_factor_text(text: str) -> pd.DataFrame:
    lines = [line.rstrip() for line in text.splitlines()]
    header_idx = next(
        (idx for idx, line in enumerate(lines) if "Mkt-RF" in line and "SMB" in line and "CMA" in line),
        None,
    )
    if header_idx is None:
        raise ValueError("Could not find five-factor header row.")

    headers = [col.strip() for col in lines[header_idx].split(",")]
    rows = _collect_numeric_rows(lines, header_idx + 1)
    if not rows:
        raise ValueError("No factor rows were found.")

    width = len(headers)
    rows = [row[:width] for row in rows]
    df = pd.DataFrame(rows, columns=headers)
    date_col = headers[0]
    df[date_col] = _build_date_index(df[date_col])
    df.set_index(date_col, inplace=True)
    return _clean_numeric_frame(df)


def parse_portfolio_text(text: str, section_hint: str) -> pd.DataFrame:
    lines = [line.rstrip() for line in text.splitlines()]
    section_idx = next((idx for idx, line in enumerate(lines) if section_hint in line), None)
    if section_idx is None:
        raise ValueError(f"Could not find section '{section_hint}'.")

    header_idx = next((idx for idx in range(section_idx + 1, len(lines)) if lines[idx].startswith(",")), None)
    if header_idx is None:
        raise ValueError("Could not find portfolio header row.")

    headers = [col.strip() for col in lines[header_idx].split(",")]
    rows = _collect_numeric_rows(lines, header_idx + 1)
    if not rows:
        raise ValueError("No portfolio rows were found.")

    width = len(headers)
    rows = [row[:width] for row in rows]
    df = pd.DataFrame(rows, columns=headers)
    date_col = headers[0]
    df[date_col] = _build_date_index(df[date_col])
    df.set_index(date_col, inplace=True)
    df = _clean_numeric_frame(df)

    if len(str(rows[0][0]).strip()) == 8:
        # Daily portfolio returns are stored in percent units; compound to monthly percent returns.
        daily_decimal = df / 100.0
        monthly_decimal = (1.0 + daily_decimal).resample("MS").prod() - 1.0
        df = monthly_decimal * 100.0

    return df


def load_factor_dataset(source_name: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    config = FACTOR_DATASETS[source_name]
    text = _read_text_from_path(config["filename"])
    if text is None:
        text = _read_remote_zip_text(config["remote_zip_url"])
    df = parse_five_factor_text(text)
    if start_date or end_date:
        df = df.loc[start_date:end_date]
    return df


def load_portfolio_dataset(source_name: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    config = PORTFOLIO_DATASETS[source_name]
    if config is None:
        return pd.DataFrame()

    text = _read_text_from_path(config["filename"])
    if text is None:
        raise FileNotFoundError(f"Portfolio dataset '{source_name}' was not found in repo or Downloads.")

    df = parse_portfolio_text(text, section_hint=config["section_hint"])
    if start_date or end_date:
        df = df.loc[start_date:end_date]
    return df
