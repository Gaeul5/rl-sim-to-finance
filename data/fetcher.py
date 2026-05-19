import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """yf.Ticker().history() 방식으로 OHLCV 수집 — yf.download()보다 API 변경에 강함."""
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for '{ticker}' ({start}~{end}). 티커 확인 또는 잠시 후 재시도하세요.")
    df = df[["Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # tz 제거로 downstream 호환
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
