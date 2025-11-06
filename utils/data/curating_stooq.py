from __future__ import annotations  # we enable postponed evaluation of annotations for forward refs
from pathlib import Path  # we import path handling
from typing import Dict, List, Literal, Optional, Sequence, Union  # we import precise typing
import pandas as pd  # we import pandas


__all__ = [
    "stooq_txt_to_df",
    "list_txt_files",
    "curate_stooq_many",
    "curate_stooq_dir",
    "stooq_txt_to_df_hourly",
    "stooq_txt_to_df_5min",
    "curate_stooq_many_hourly",
    "curate_stooq_dir_hourly",
    "curate_stooq_many_5min",
    "curate_stooq_dir_5min",
]  # we expose public api of this module


def stooq_txt_to_df(file_path: Union[str, Path], tz: Optional[str] = None) -> pd.DataFrame:
    '''
    we parse a stooq txt file into a pandas time series dataframe with a datetime index  # we describe the purpose
    we expect columns: <ticker>,<per>,<date>,<time>,<open>,<high>,<low>,<close>,<vol>,<openint>  # we state expected format
    we support trailing commas by reading only the first 10 columns  # we mention robustness

    args:
        file_path: path to the stooq txt file  # we document parameter
        tz: optional timezone name to localize the datetime index (e.g., "UTC" or "Europe/Paris")  # we document parameter

    returns:
        a dataframe indexed by datetime with columns: ticker, per, open, high, low, close, volume, openint  # we describe return

    raises:
        filenotfounderror: if file_path does not exist  # we document errors
        valueerror: if no valid rows remain after parsing  # we document errors
    '''
    p = Path(file_path)  # we normalize to a path object
    if not p.is_file():  # we validate the file exists
        raise FileNotFoundError(f"file not found: {p}")  # we raise a clear error

    names = [
        "ticker",
        "per",
        "date",
        "time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "openint",
    ]  # we define column names

    dtypes = {
        "ticker": "string",
        "per": "string",
        "date": "string",
        "time": "string",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "vol": "Int64",
        "openint": "Int64",
        }  # we set explicit dtypes for reliability
    
    
    df = pd.read_csv(
        p,  # we pass path
        header=None,  # we specify no header
        names=names,  # we set column names
        sep=",",  # we set csv-like separator
        engine="python",  # we use python engine to allow on_bad_lines
        comment="<",  # we skip header lines like <TICKER>,<PER>,... safely
        usecols=list(range(10)),  # we read only first 10 cols
        skip_blank_lines=True,  # we skip blank lines
        dtype=dtypes,  # we enforce dtypes
        na_filter=True,  # we enable na parsing
        encoding="utf-8",  # we set encoding
        on_bad_lines="skip"  # we skip malformed lines
    )  # we load file robustly and ignore any extra trailing columns
    if df.empty:  # we ensure not empty
        raise ValueError(f"no rows read from file: {p}")  # we signal an error

    df["time"] = df["time"].fillna("000000").astype("string").str.zfill(6)  # we normalize time to HHMMSS
    dt = pd.to_datetime(
        df["date"].astype("string") + df["time"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
        utc=False,
    )  # we build a timestamp from date and time

    df = df.assign(timestamp=dt)  # we add the timestamp column
    df = df.dropna(subset=["timestamp"])  # we drop rows with invalid datetime

    if df.empty:  # we recheck after cleaning
        raise ValueError(f"no valid datetime rows remain after parsing: {p}")  # we report invalid file content

    df = df.rename(columns={"vol": "volume"})  # we standardize volume name
    df = df.drop(columns=["date", "time"])  # we drop raw date/time after conversion
    df = df.set_index("timestamp").sort_index()  # we set the datetime index and sort
    df = df[~df.index.duplicated(keep="last")]  # we drop duplicate timestamps keeping the last occurrence

    if tz is not None:  # we localize timezone if requested
        df.index = df.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")  # we localize with safe rules

    # we keep a clean column order for readability
    cols = ["ticker", "per", "open", "high", "low", "close", "volume", "openint"]  # we define the order
    existing = [c for c in cols if c in df.columns]  # we filter to existing columns
    df = df[existing]  # we reorder columns

    return df  # we return the curated dataframe


def list_txt_files(dir_path: Union[str, Path], pattern: str = "*.txt", recursive: bool = True) -> List[Path]:
    '''
    we list all txt files under a directory optionally recursively  # we describe the purpose

    args:
        dir_path: directory to search  # we document parameter
        pattern: filename glob pattern (default: "*.txt")  # we document parameter
        recursive: whether to search subdirectories (default: True)  # we document parameter

    returns:
        a sorted list of pathlib paths pointing to files  # we describe return

    raises:
        filenotfounderror: if dir_path does not exist  # we document errors
    '''
    d = Path(dir_path)  # we normalize to a path
    if not d.exists():  # we check folder existence
        raise FileNotFoundError(f"directory not found: {d}")  # we raise if missing

    paths = d.rglob(pattern) if recursive else d.glob(pattern)  # we enumerate files
    files = [p for p in paths if p.is_file()]  # we keep only files
    files = sorted(files)  # we sort deterministically

    return files  # we return file list


def curate_stooq_many(
    paths: Sequence[Union[str, Path]],
    tz: Optional[str] = None,
    errors: Literal["raise", "skip"] = "skip",
) -> Dict[str, pd.DataFrame]:
    '''
    we apply stooq_txt_to_df to each path and collect results in a dict keyed by ticker  # we describe the purpose

    args:
        paths: iterable of file paths  # we document parameter
        tz: optional timezone name to localize indices  # we document parameter
        errors: "skip" to continue on errors, "raise" to propagate  # we document parameter

    returns:
        a dict: {ticker -> dataframe}  # we describe return
    '''
    out: Dict[str, pd.DataFrame] = {}  # we prepare the output dict

    for path in paths:  # we iterate over files
        try:
            df = stooq_txt_to_df(path, tz=tz)  # we transform one file
            if df.empty:  # we guard against empty
                continue  # we skip empty
            ticker = str(df["ticker"].iloc[0])  # we take ticker as the key
            out[ticker] = df  # we store the dataframe by ticker
        except Exception as exc:  # we handle per-file errors
            if errors == "raise":  # we propagate if requested
                raise  # we re-raise the exception
            # else we skip silently  # we skip on error

    return out  # we return the mapping


def curate_stooq_dir(
    dir_path: Union[str, Path],
    pattern: str = "*.txt",
    recursive: bool = True,
    tz: Optional[str] = None,
    errors: Literal["raise", "skip"] = "skip",
) -> Dict[str, pd.DataFrame]:
    '''
    we list txt files in a directory and curate them into dataframes keyed by ticker  # we describe the purpose

    args:
        dir_path: directory to search  # we document parameter
        pattern: filename glob pattern (default: "*.txt")  # we document parameter
        recursive: whether to search subdirectories (default: True)  # we document parameter
        tz: optional timezone name to localize indices  # we document parameter
        errors: "skip" to continue on errors, "raise" to propagate  # we document parameter

    returns:
        a dict: {ticker -> dataframe}  # we describe return
    '''
    files = list_txt_files(dir_path, pattern=pattern, recursive=recursive)  # we find candidate files
    return curate_stooq_many(files, tz=tz, errors=errors)  # we parse each file and collect results


def stooq_txt_to_df_hourly(file_path: Union[str, Path], tz: Optional[str] = None) -> pd.DataFrame:
    '''
    we parse a stooq txt file and return only hourly (60-minute) bars with validated timestamps  # we describe the purpose

    args:
        file_path: path to stooq txt  # we document parameter
        tz: optional timezone to localize index  # we document parameter

    returns:
        a dataframe filtered to per=="60" with timestamps at hh:00:00  # we describe return
    '''
    df = stooq_txt_to_df(file_path, tz=tz)  # we parse base format
    if "per" not in df.columns:  # we ensure per column exists
        return df.iloc[0:0]  # we return empty if missing
    mask = df["per"].astype("string").str.upper() == "60"  # we select hourly rows
    df = df.loc[mask]  # we filter to hourly
    if df.empty:  # we handle empty result
        return df  # we return empty dataframe
    valid = (df.index.minute == 0) & (df.index.second == 0)  # we validate timestamps align to the hour
    df = df.loc[valid]  # we keep only valid rows
    df = df[~df.index.duplicated(keep="last")]  # we deduplicate again for safety
    return df  # we return hourly dataframe


def stooq_txt_to_df_5min(file_path: Union[str, Path], tz: Optional[str] = None) -> pd.DataFrame:
    '''
    we parse a stooq txt file and return only 5-minute bars with validated timestamps  # we describe the purpose

    args:
        file_path: path to stooq txt  # we document parameter
        tz: optional timezone to localize index  # we document parameter

    returns:
        a dataframe filtered to per=="5" with timestamps at multiples of 5 minutes  # we describe return
    '''
    df = stooq_txt_to_df(file_path, tz=tz)  # we parse base format
    if "per" not in df.columns:  # we ensure per column exists
        return df.iloc[0:0]  # we return empty if missing
    mask = df["per"].astype("string").str.upper() == "5"  # we select 5-minute rows
    df = df.loc[mask]  # we filter to 5min
    if df.empty:  # we handle empty result
        return df  # we return empty dataframe
    valid = (df.index.second == 0) & ((df.index.minute % 5) == 0)  # we validate timestamps align to 5-minute grid
    df = df.loc[valid]  # we keep only valid rows
    df = df[~df.index.duplicated(keep="last")]  # we deduplicate again for safety
    return df  # we return 5-minute dataframe


def curate_stooq_many_hourly(
    paths: Sequence[Union[str, Path]],
    tz: Optional[str] = None,
    errors: Literal["raise", "skip"] = "skip",
) -> Dict[str, pd.DataFrame]:
    '''
    we parse multiple files keeping only hourly (60-minute) bars  # we describe the purpose
    '''
    out: Dict[str, pd.DataFrame] = {}  # we prepare output mapping
    for path in paths:  # we iterate files
        try:
            df = stooq_txt_to_df_hourly(path, tz=tz)  # we parse hourly
            if df.empty:  # we skip empty
                continue  # we continue
            ticker = str(df["ticker"].iloc[0])  # we key by ticker
            out[ticker] = df  # we store hourly df
        except Exception as exc:  # we handle per-file errors
            if errors == "raise":  # we propagate if requested
                raise  # we re-raise
            # else we skip  # we skip on error
    return out  # we return mapping


def curate_stooq_dir_hourly(
    dir_path: Union[str, Path],
    pattern: str = "*.txt",
    recursive: bool = True,
    tz: Optional[str] = None,
    errors: Literal["raise", "skip"] = "skip",
) -> Dict[str, pd.DataFrame]:
    '''
    we list files and parse only hourly (60-minute) bars  # we describe the purpose
    '''
    files = list_txt_files(dir_path, pattern=pattern, recursive=recursive)  # we list files
    return curate_stooq_many_hourly(files, tz=tz, errors=errors)  # we parse and collect


def curate_stooq_many_5min(
    paths: Sequence[Union[str, Path]],
    tz: Optional[str] = None,
    errors: Literal["raise", "skip"] = "skip",
) -> Dict[str, pd.DataFrame]:
    '''
    we parse multiple files keeping only 5-minute bars  # we describe the purpose
    '''
    out: Dict[str, pd.DataFrame] = {}  # we prepare output mapping
    for path in paths:  # we iterate files
        try:
            df = stooq_txt_to_df_5min(path, tz=tz)  # we parse 5-minute
            if df.empty:  # we skip empty
                continue  # we continue
            ticker = str(df["ticker"].iloc[0])  # we key by ticker
            out[ticker] = df  # we store 5-minute df
        except Exception as exc:  # we handle per-file errors
            if errors == "raise":  # we propagate if requested
                raise  # we re-raise
            # else we skip  # we skip on error
    return out  # we return mapping


def curate_stooq_dir_5min(
    dir_path: Union[str, Path],
    pattern: str = "*.txt",
    recursive: bool = True,
    tz: Optional[str] = None,
    errors: Literal["raise", "skip"] = "skip",
) -> Dict[str, pd.DataFrame]:
    '''
    we list files and parse only 5-minute bars  # we describe the purpose
    '''
    files = list_txt_files(dir_path, pattern=pattern, recursive=recursive)  # we list files
    return curate_stooq_many_5min(files, tz=tz, errors=errors)  # we parse and collect