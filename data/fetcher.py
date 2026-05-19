import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and return a clean DataFrame with Close/Volume."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' ({start} ~ {end})")
    df = df[["Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def add_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Append rolling-mean and rolling-std features used by the trading env."""
    df = df.copy()
    df["MA"] = df["Close"].rolling(window).mean()
    df["Std"] = df["Close"].rolling(window).std()
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    return df


def normalize_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Min-max normalize Close price; return scaled df and scale factor."""
    scale = float(df["Close"].max())
    df = df.copy()
    df["Close"] = df["Close"] / scale
    return df, scale


def train_test_split_df(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()
