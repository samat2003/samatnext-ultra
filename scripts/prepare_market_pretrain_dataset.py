#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP005: Binance Futures Market Pretraining Dataset preparation script.

Builds a token-ready market-state pretraining corpus from Binance public
historical USD(S)-M futures monthly kline data.
"""

import argparse
import datetime
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

# Define token offset ranges
TOKEN_SPECIAL_PAD = 0
TOKEN_SPECIAL_BOS = 1
TOKEN_SPECIAL_EOS = 2
TOKEN_SPECIAL_SEP = 3
TOKEN_SPECIAL_MASK = 4

RANGE_SYMBOL = (5, 8)          # 4 symbols (BTCUSDT=5, ETHUSDT=6, SOLUSDT=7, BNBUSDT=8)
RANGE_INTERVAL = (9, 11)        # 3 intervals (1m=9, 5m=10, 15m=11)
RANGE_TIME_OF_DAY = (12, 35)    # 24 buckets
RANGE_DAY_OF_WEEK = (36, 42)    # 7 buckets
RANGE_CC_RETURN = (43, 72)      # 30 buckets
RANGE_OC_RETURN = (73, 102)     # 30 buckets
RANGE_HL_RANGE = (103, 132)     # 30 buckets
RANGE_VOLUME = (133, 162)       # 30 buckets
RANGE_QUOTE_VOLUME = (163, 192) # 30 buckets
RANGE_TRADE_COUNT = (193, 222)  # 30 buckets
RANGE_IMBALANCE = (223, 242)    # 20 buckets
RANGE_VOLATILITY = (243, 255)   # 13 buckets

SYMBOL_MAP = {
    "BTCUSDT": 5,
    "ETHUSDT": 6,
    "SOLUSDT": 7,
    "BNBUSDT": 8,
}
INTERVAL_MAP = {
    "1m": 9,
    "5m": 10,
    "15m": 11,
}

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def parse_date(date_str: str) -> datetime.date:
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def generate_months(start_date: datetime.date, end_date: datetime.date) -> list[str]:
    months = []
    curr = start_date.replace(day=1)
    while curr <= end_date:
        months.append(curr.strftime("%Y-%m"))
        # advance to next month
        if curr.month == 12:
            curr = curr.replace(year=curr.year + 1, month=1)
        else:
            curr = curr.replace(month=curr.month + 1)
    return months


def download_file(url: str, dest_path: Path, force: bool = False) -> bool:
    if dest_path.exists() and not force:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {url} ...")
        # simple download helper with 15s timeout
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            dest_path.write_bytes(response.read())
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


COLUMN_RENAME_MAP = {
    "quote_volume": "quote_asset_volume",
    "count": "number_of_trades",
    "taker_buy_volume": "taker_buy_base_asset_volume",
    "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
}

def load_kline_csv(zip_path: Path, symbol: str, interval: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        matching = [n for n in names if n.endswith(".csv")]
        if not matching:
            raise ValueError(f"No CSV found in zip {zip_path}")
        
        with z.open(matching[0]) as f:
            first_line = f.readline().decode("utf-8")
            f.seek(0)
            has_header = "open_time" in first_line or "open" in first_line
            
            if has_header:
                df = pd.read_csv(f, header=0)
                df = df.rename(columns=COLUMN_RENAME_MAP)
            else:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
            
            df["symbol"] = symbol
            df["interval"] = interval
            return df


def validate_and_clean_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Applies Binance-specific data validation constraints."""
    total_rows = len(df)
    
    # 1. Invalid Row Check (non-positive OHLC, bounds logic, negative volumes)
    valid_mask = (
        (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) &
        (df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
        (df["low"] <= df[["open", "close", "high"]].min(axis=1)) &
        (df["volume"] >= 0) &
        (df["quote_asset_volume"] >= 0) &
        (df["number_of_trades"] >= 0) &
        (df["taker_buy_base_asset_volume"] >= 0) &
        (df["taker_buy_quote_asset_volume"] >= 0)
    )
    
    df_clean = df[valid_mask].copy()
    invalid_count = total_rows - len(df_clean)
    
    # 2. Duplicate Check
    df_sorted = df_clean.sort_values(by="open_time")
    duplicate_mask = df_sorted.duplicated(subset=["open_time"], keep="first")
    df_dedup = df_sorted[~duplicate_mask].copy()
    duplicate_count = len(df_sorted) - len(df_dedup)
    
    return df_dedup, invalid_count, duplicate_count


def compute_feature_bins(train_values: np.ndarray, num_bins: int) -> np.ndarray:
    """Computes bin edges using train split quantiles only."""
    # Ensure values are valid and finite
    clean_vals = train_values[np.isfinite(train_values)]
    if len(clean_vals) == 0:
        return np.linspace(-1e-5, 1e-5, num_bins - 1)
        
    percentiles = np.linspace(0, 100, num_bins + 1)[1:-1]
    edges = np.percentile(clean_vals, percentiles)
    # Ensure strictly increasing
    edges = np.unique(edges)
    if len(edges) < num_bins - 1:
        # Fallback if too many duplicate values (e.g. constant returns or volumes)
        eps = np.linspace(1e-9, 1e-7, num_bins - 1 - len(edges))
        edges = np.sort(np.concatenate([edges, eps]))
    return edges


def quantize_feature(values: pd.Series, edges: np.ndarray, range_min: int, range_max: int) -> np.ndarray:
    """Maps continuous feature series to dedicated token range buckets."""
    num_buckets = range_max - range_min + 1
    # numpy.digitize maps values <= edges[0] to 0, values > edges[-1] to len(edges)
    idxs = np.digitize(values.values, edges)
    # clip to valid index range just in case
    idxs = np.clip(idxs, 0, num_buckets - 1)
    return idxs + range_min


def process_features_and_tokenize(
    df: pd.DataFrame,
    is_train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]], dict[str, int]]:
    """Calculates normalized continuous features, sets up boundaries from train split, and tokenizes."""
    # Ensure correct sort order before feature calculations
    df = df.sort_values(by=["symbol", "open_time"]).copy()
    
    # Feature 1: close-to-close log return
    df["cc_return"] = np.log(df["close"]) - np.log(df.groupby("symbol")["close"].shift(1))
    
    # Feature 2: open-to-close log return
    df["oc_return"] = np.log(df["close"]) - np.log(df["open"])
    
    # Feature 3: high-low range
    df["hl_range"] = np.log(df["high"]) - np.log(df["low"])
    
    # Feature 4: taker buy volume imbalance
    # Avoid div-by-zero, clip to [0, 1]
    df["imbalance"] = (df["taker_buy_base_asset_volume"] / df["volume"].replace(0, 1e-8)).clip(0.0, 1.0)
    
    # Feature 5: realized volatility (rolling std of past 30 CC returns, shifted by 1 to prevent leakage)
    df["realized_vol"] = df.groupby("symbol")["cc_return"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=30).std()
    )
    
    # Fill missing values for boundary first bars
    # "For first rows where prior data is unavailable, use neutral buckets and document the rule."
    # We will fill NaNs with a unique placeholder, then assign the median bucket index
    df["cc_return"] = df["cc_return"].fillna(0.0)
    df["oc_return"] = df["oc_return"].fillna(0.0)
    df["hl_range"] = df["hl_range"].fillna(0.0)
    df["imbalance"] = df["imbalance"].fillna(0.5)
    
    # Extract training subset for boundary calculations
    train_df = df[is_train_mask]
    
    # Compute bin boundaries
    bins_cc = compute_feature_bins(train_df["cc_return"].values, 30)
    bins_oc = compute_feature_bins(train_df["oc_return"].values, 30)
    bins_hl = compute_feature_bins(train_df["hl_range"].values, 30)
    bins_vol = compute_feature_bins(train_df["volume"].values, 30)
    bins_quote = compute_feature_bins(train_df["quote_asset_volume"].values, 30)
    bins_trades = compute_feature_bins(train_df["number_of_trades"].values, 30)
    bins_imb = compute_feature_bins(train_df["imbalance"].values, 20)
    
    # Realized volatility training subset
    train_vols = train_df["realized_vol"].dropna().values
    bins_rv = compute_feature_bins(train_vols, 13)
    
    # Save boundaries for metadata (convert to lists for JSON compatibility)
    boundaries = {
        "cc_return": bins_cc.tolist(),
        "oc_return": bins_oc.tolist(),
        "hl_range": bins_hl.tolist(),
        "volume": bins_vol.tolist(),
        "quote_volume": bins_quote.tolist(),
        "number_of_trades": bins_trades.tolist(),
        "imbalance": bins_imb.tolist(),
        "realized_volatility": bins_rv.tolist(),
    }
    
    # Now quantize all rows
    # Default placeholder fills for missing volatility values (first 30 rows)
    vol_neutral_token = (RANGE_VOLATILITY[1] - RANGE_VOLATILITY[0]) // 2 + RANGE_VOLATILITY[0]
    
    tok_symbol = df["symbol"].map(SYMBOL_MAP).values
    tok_interval = df["interval"].map(INTERVAL_MAP).values
    
    # Time of day (0-23) -> range 12-35
    times = pd.to_datetime(df["open_time"], unit="ms")
    tok_time = (times.dt.hour.values) + RANGE_TIME_OF_DAY[0]
    
    # Day of week (0-6) -> range 36-42
    tok_dow = (times.dt.weekday.values) + RANGE_DAY_OF_WEEK[0]
    
    tok_cc = quantize_feature(df["cc_return"], bins_cc, *RANGE_CC_RETURN)
    tok_oc = quantize_feature(df["oc_return"], bins_oc, *RANGE_OC_RETURN)
    tok_hl = quantize_feature(df["hl_range"], bins_hl, *RANGE_HL_RANGE)
    tok_vol = quantize_feature(df["volume"], bins_vol, *RANGE_VOLUME)
    tok_quote = quantize_feature(df["quote_asset_volume"], bins_quote, *RANGE_QUOTE_VOLUME)
    tok_trades = quantize_feature(df["number_of_trades"], bins_trades, *RANGE_TRADE_COUNT)
    tok_imb = quantize_feature(df["imbalance"], bins_imb, *RANGE_IMBALANCE)
    
    # Volatility needs special neutral bucket fill where rolling data is missing
    vol_mask = df["realized_vol"].isna().values
    tok_rv = np.zeros(len(df), dtype=np.int32)
    tok_rv[vol_mask] = vol_neutral_token
    tok_rv[~vol_mask] = quantize_feature(df["realized_vol"].dropna(), bins_rv, *RANGE_VOLATILITY)
    
    # Record overflow/clipping counts
    # Overflow occurs when values lie outside the computed boundaries
    clipping_counts = {
        "cc_return_clipped_low": int(np.sum(df["cc_return"].values < bins_cc[0])),
        "cc_return_clipped_high": int(np.sum(df["cc_return"].values > bins_cc[-1])),
        "oc_return_clipped_low": int(np.sum(df["oc_return"].values < bins_oc[0])),
        "oc_return_clipped_high": int(np.sum(df["oc_return"].values > bins_oc[-1])),
        "hl_range_clipped_low": int(np.sum(df["hl_range"].values < bins_hl[0])),
        "hl_range_clipped_high": int(np.sum(df["hl_range"].values > bins_hl[-1])),
        "volume_clipped_low": int(np.sum(df["volume"].values < bins_vol[0])),
        "volume_clipped_high": int(np.sum(df["volume"].values > bins_vol[-1])),
        "quote_volume_clipped_low": int(np.sum(df["quote_asset_volume"].values < bins_quote[0])),
        "quote_volume_clipped_high": int(np.sum(df["quote_asset_volume"].values > bins_quote[-1])),
        "trade_count_clipped_low": int(np.sum(df["number_of_trades"].values < bins_trades[0])),
        "trade_count_clipped_high": int(np.sum(df["number_of_trades"].values > bins_trades[-1])),
        "imbalance_clipped_low": int(np.sum(df["imbalance"].values < bins_imb[0])),
        "imbalance_clipped_high": int(np.sum(df["imbalance"].values > bins_imb[-1])),
        "realized_volatility_clipped_low": int(np.sum(df["realized_vol"].dropna().values < bins_rv[0])),
        "realized_volatility_clipped_high": int(np.sum(df["realized_vol"].dropna().values > bins_rv[-1])),
    }
    
    # Build token representation array: 14 elements per bar
    # [BOS, symbol_id, interval_id, time_bucket, dow_bucket, return_bucket, oc_return_bucket, range_bucket, volume_bucket, quote_volume_bucket, trade_count_bucket, imbalance_bucket, volatility_bucket, SEP]
    bos = np.full(len(df), TOKEN_SPECIAL_BOS, dtype=np.uint8)
    sep = np.full(len(df), TOKEN_SPECIAL_SEP, dtype=np.uint8)
    
    tokens = np.stack([
        bos,
        tok_symbol.astype(np.uint8),
        tok_interval.astype(np.uint8),
        tok_time.astype(np.uint8),
        tok_dow.astype(np.uint8),
        tok_cc.astype(np.uint8),
        tok_oc.astype(np.uint8),
        tok_hl.astype(np.uint8),
        tok_vol.astype(np.uint8),
        tok_quote.astype(np.uint8),
        tok_trades.astype(np.uint8),
        tok_imb.astype(np.uint8),
        tok_rv.astype(np.uint8),
        sep
    ], axis=1).reshape(-1)
    
    # Keep some metrics attached to df for sample previews
    df["token_representation"] = [list(toks) for toks in tokens.reshape(-1, 14)]
    
    return tokens, boundaries, clipping_counts, df


def write_binary_file(tokens: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(path)


def generate_sample_preview(df: pd.DataFrame, symbols: list[str]) -> dict:
    preview = {}
    for symbol in symbols:
        sym_df = df[df["symbol"] == symbol].head(5)
        bars_preview = []
        for _, row in sym_df.iterrows():
            bars_preview.append({
                "open_time": int(row["open_time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "quote_asset_volume": float(row["quote_asset_volume"]),
                "number_of_trades": int(row["number_of_trades"]),
                "taker_buy_base_asset_volume": float(row["taker_buy_base_asset_volume"]),
                "taker_buy_quote_asset_volume": float(row["taker_buy_quote_asset_volume"]),
                "cc_return": float(row["cc_return"]),
                "oc_return": float(row["oc_return"]),
                "hl_range": float(row["hl_range"]),
                "imbalance": float(row["imbalance"]),
                "realized_volatility": float(row["realized_vol"]) if pd.notna(row["realized_vol"]) else None,
                "token_ids": [int(t) for t in row["token_representation"]],
            })
        preview[symbol] = bars_preview
    return preview


def create_dataset_card(out_dir: Path, exact_dates: str, prepared_full: bool) -> None:
    card_path = out_dir / "DATASET_CARD.md"
    content = f"""# Dataset Card: Binance Futures Market Pretraining Dataset

## Dataset Description
- **Dataset Name:** Binance UM Futures 1m Pretraining Dataset
- **Market Type:** USDⓈ-M Futures
- **Symbols:** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`
- **Interval:** `1m`
- **Date Range:** {exact_dates}
- **Scale Status:** {"Full 12-Month Dataset" if prepared_full else "30-Day Smoke Subset (Fallback)"}

## Preprocessing & Data Cleaning
- Sort by Symbol and Open Time.
- Duplicate bars removed.
- Missing bar detection.
- Validation checks applied:
  - Drop rows with non-positive values of open, high, low, close.
  - Drop rows where `high < max(open, close, low)`.
  - Drop rows where `low > min(open, close, high)`.
  - Drop rows with negative volume, quote asset volume, number of trades, or taker buy volumes.
  - **Important:** Zero-volume rows are preserved if they are otherwise structurally valid.

## Tokenization Scheme
Each kline is tokenized into a structured sequence of 14 byte-compatible integer tokens in `[0, 255]`:
`[BOS, symbol_id, interval_id, time_bucket, dow_bucket, cc_return_bucket, oc_return_bucket, range_bucket, volume_bucket, quote_volume_bucket, trade_count_bucket, imbalance_bucket, volatility_bucket, SEP]`

### Dedicated Token Ranges:
- `0 - 4`: Special Tokens (`0=PAD`, `1=BOS`, `2=EOS`, `3=SEP`, `4=MASK`)
- `5 - 8`: Symbol Identifier (`BTCUSDT=5`, `ETHUSDT=6`, `SOLUSDT=7`, `BNBUSDT=8`)
- `9 - 11`: Interval Identifier (`1m=9`, `5m=10`, `15m=11`)
- `12 - 35`: Time-of-day Bucket (24 hourly buckets)
- `36 - 42`: Day-of-week Bucket (7 day-of-week buckets)
- `43 - 72`: Close-to-Close Log Return Bucket (30 buckets)
- `73 - 102`: Open-to-Close Return Bucket (30 buckets)
- `103 - 132`: High-Low Range Bucket (30 buckets)
- `133 - 162`: Volume Bucket (30 buckets)
- `163 - 192`: Quote Volume Bucket (30 buckets)
- `193 - 222`: Trade Count Bucket (30 buckets)
- `223 - 242`: Taker Buy Imbalance Bucket (20 buckets)
- `243 - 255`: Realized Volatility Bucket (13 buckets)

## Split Method (Chronological)
To prevent lookahead and statistical leakage, the dataset is split globally and chronologically based on sorted timestamps:
- **Train:** Earliest 80% of unique timestamps
- **Validation:** Next 10% of unique timestamps
- **Test:** Final 10% of unique timestamps

Quantization boundary buckets are computed strictly using the training split and applied identically to validation and test sets.

---

## Disclaimers & Notes
- **No Trading Decision Labels:** This pretraining dataset is for next-token structure learning only. It does NOT contain `LONG`, `SHORT`, `HOLD`, or other decision/profit labels.
- **Not Financial Advice:** This artifact is built strictly for AI research purposes. Do not use this data or model projections for active live trading.
"""
    card_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Binance Futures Pretraining Dataset")
    parser.add_argument("--source", default="binance_um_futures")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start-date", default="2025-06-01")
    parser.add_argument("--end-date", default="2026-05-31")
    parser.add_argument("--out-dir", default="data/market_pretrain/binance_um_futures_1m")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    months = generate_months(start_date, end_date)
    
    downloaded_files = []
    source_urls = []
    
    # 1. Download Mode
    if args.download:
        print(f"Downloading files for {len(args.symbols)} symbols over {len(months)} months...")
        success_count = 0
        total_expected = len(args.symbols) * len(months)
        
        for symbol in args.symbols:
            for month in months:
                url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{args.interval}/{symbol}-{args.interval}-{month}.zip"
                dest = raw_dir / f"{symbol}-{args.interval}-{month}.zip"
                if download_file(url, dest, force=args.force):
                    downloaded_files.append(str(dest.relative_to(out_dir)))
                    source_urls.append(url)
                    success_count += 1
                else:
                    print(f"Failed to download monthly file for {symbol} {month}.")
        
        # Check if we prepared full or fall back to 30 days
        if success_count < total_expected:
            print(f"\n[Warning] Could only download {success_count}/{total_expected} files.")
            print("Preparing a 30-day smoke subset (e.g. 2025-06) as fallback...")
            # Set to 2025-06 only
            months = ["2025-06"]
            downloaded_files = []
            source_urls = []
            for symbol in args.symbols:
                url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{args.interval}/{symbol}-{args.interval}-2025-06.zip"
                dest = raw_dir / f"{symbol}-{args.interval}-2025-06.zip"
                if download_file(url, dest, force=args.force):
                    downloaded_files.append(str(dest.relative_to(out_dir)))
                    source_urls.append(url)
    
    # 2. Process Mode
    if args.process:
        print("\nIngesting and parsing downloaded files...")
        combined_dfs = []
        
        missing_bars_per_symbol = {}
        invalid_rows_per_symbol = {}
        duplicate_rows_per_symbol = {}
        rows_per_symbol = {}
        
        for symbol in args.symbols:
            symbol_dfs = []
            for month in months:
                zip_path = raw_dir / f"{symbol}-{args.interval}-{month}.zip"
                if not zip_path.exists():
                    continue
                try:
                    df_month = load_kline_csv(zip_path, symbol, args.interval)
                    symbol_dfs.append(df_month)
                except Exception as e:
                    print(f"Error loading {zip_path}: {e}")
                    
            if not symbol_dfs:
                print(f"No valid data found for symbol {symbol}.")
                continue
                
            symbol_df = pd.concat(symbol_dfs, ignore_index=True)
            
            # Save raw count
            raw_count = len(symbol_df)
            
            # Clean and validate row schema
            clean_df, invalid_cnt, duplicate_cnt = validate_and_clean_rows(symbol_df)
            
            # Missing Bar detection
            # 1m interval means open_time is spaced by 60,000 ms
            # Check spacing
            open_times = clean_df["open_time"].values
            if len(open_times) > 1:
                diffs = np.diff(open_times)
                # expected difference is 60,000 ms for 1m
                expected_diff = 60_000
                missing_cnt = int(np.sum((diffs > expected_diff) * ((diffs / expected_diff) - 1)))
            else:
                missing_cnt = 0
                
            missing_bars_per_symbol[symbol] = missing_cnt
            invalid_rows_per_symbol[symbol] = invalid_cnt
            duplicate_rows_per_symbol[symbol] = duplicate_cnt
            rows_per_symbol[symbol] = len(clean_df)
            
            combined_dfs.append(clean_df)
            
        if not combined_dfs:
            print("No data processed. Exiting.")
            sys.exit(1)
            
        full_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Sort globally by open_time
        full_df = full_df.sort_values(by="open_time").reset_index(drop=True)
        
        # Global chronological split
        unique_times = np.sort(full_df["open_time"].unique())
        total_times = len(unique_times)
        
        train_idx_max = int(total_times * 0.8)
        val_idx_max = int(total_times * 0.9)
        
        train_t_max = unique_times[train_idx_max - 1]
        val_t_max = unique_times[val_idx_max - 1]
        
        train_mask = (full_df["open_time"] <= train_t_max).values
        val_mask = ((full_df["open_time"] > train_t_max) & (full_df["open_time"] <= val_t_max)).values
        test_mask = (full_df["open_time"] > val_t_max).values
        
        # Row counts inside each split per symbol
        per_symbol_split_counts = {}
        for symbol in args.symbols:
            sym_mask = (full_df["symbol"] == symbol).values
            per_symbol_split_counts[symbol] = {
                "train_rows": int(np.sum(sym_mask & train_mask)),
                "val_rows": int(np.sum(sym_mask & val_mask)),
                "test_rows": int(np.sum(sym_mask & test_mask)),
            }
            
        # Calculate features and tokenize
        print("Calculating normalized features and tokenizing...")
        tokens, boundaries, clipping_counts, full_df = process_features_and_tokenize(full_df, train_mask)
        
        # Tokens array is aligned with full_df rows: each row represents exactly 14 tokens
        tokens_reshaped = tokens.reshape(-1, 14)
        
        train_tokens = tokens_reshaped[train_mask].reshape(-1)
        val_tokens = tokens_reshaped[val_mask].reshape(-1)
        test_tokens = tokens_reshaped[test_mask].reshape(-1)
        
        # Save binaries
        write_binary_file(train_tokens, out_dir / "train.bin")
        write_binary_file(val_tokens, out_dir / "val.bin")
        write_binary_file(test_tokens, out_dir / "test.bin")
        
        # Generate sample preview JSON
        preview = generate_sample_preview(full_df, args.symbols)
        with open(out_dir / "sample_preview.json", "w", encoding="utf-8") as f:
            json.dump(preview, f, indent=2)
            
        # Verify chronological split boundaries
        train_dates = (
            pd.to_datetime(unique_times[0], unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            pd.to_datetime(train_t_max, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
        )
        val_dates = (
            pd.to_datetime(unique_times[train_idx_max], unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            pd.to_datetime(val_t_max, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
        )
        test_dates = (
            pd.to_datetime(unique_times[val_idx_max], unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            pd.to_datetime(unique_times[-1], unit="ms").strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Write metadata.json
        metadata = {
            "dataset_name": "binance_um_futures_1m_pretrain",
            "source_root": "https://data.binance.vision/data/futures/um/monthly/klines/",
            "market_type": "USD(S)-M futures",
            "symbols": args.symbols,
            "interval": args.interval,
            "start_date": train_dates[0],
            "end_date": test_dates[1],
            "split_rule": "80% Train, 10% Val, 10% Test global chronological split",
            "train_date_range": train_dates,
            "val_date_range": val_dates,
            "test_date_range": test_dates,
            "train_token_count": len(train_tokens),
            "val_token_count": len(val_tokens),
            "test_token_count": len(test_tokens),
            "total_token_count": len(tokens),
            "vocab_size": 256,
            "special_tokens": {
                "PAD": TOKEN_SPECIAL_PAD,
                "BOS": TOKEN_SPECIAL_BOS,
                "EOS": TOKEN_SPECIAL_EOS,
                "SEP": TOKEN_SPECIAL_SEP,
                "MASK": TOKEN_SPECIAL_MASK,
            },
            "feature_token_ranges": {
                "symbol": list(RANGE_SYMBOL),
                "interval": list(RANGE_INTERVAL),
                "time_of_day": list(RANGE_TIME_OF_DAY),
                "day_of_week": list(RANGE_DAY_OF_WEEK),
                "cc_return": list(RANGE_CC_RETURN),
                "oc_return": list(RANGE_OC_RETURN),
                "hl_range": list(RANGE_HL_RANGE),
                "volume": list(RANGE_VOLUME),
                "quote_volume": list(RANGE_QUOTE_VOLUME),
                "trade_count": list(RANGE_TRADE_COUNT),
                "imbalance": list(RANGE_IMBALANCE),
                "volatility": list(RANGE_VOLATILITY),
            },
            "bucket_boundaries": boundaries,
            "rows_per_symbol": rows_per_symbol,
            "per_symbol_split_counts": per_symbol_split_counts,
            "missing_bars_per_symbol": missing_bars_per_symbol,
            "duplicate_bars_removed_per_symbol": duplicate_rows_per_symbol,
            "invalid_rows_removed_per_symbol": invalid_rows_per_symbol,
            "clipping_counts": clipping_counts,
            "raw_files": downloaded_files,
            "source_urls": source_urls,
            "processing_timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "architecture_target": "Fast32 byte-level vocab 256",
            "training_ready": True,
        }
        
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        # Create DATASET_CARD.md
        exact_dates = f"{train_dates[0]} to {test_dates[1]}"
        prepared_full = len(months) >= 12
        create_dataset_card(out_dir, exact_dates, prepared_full)
        
        print("\nPreprocessing completed successfully!")
        print(f"Train Tokens: {len(train_tokens):,}")
        print(f"Val Tokens: {len(val_tokens):,}")
        print(f"Test Tokens: {len(test_tokens):,}")
        print(f"Total Tokens: {len(tokens):,}")
        print(f"Output files written to {out_dir}/")


if __name__ == "__main__":
    main()
