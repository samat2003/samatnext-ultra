# Dataset Card: Binance Futures Market Pretraining Dataset

## Dataset Description
- **Dataset Name:** Binance UM Futures 1m Pretraining Dataset
- **Market Type:** USDⓈ-M Futures
- **Symbols:** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`
- **Interval:** `1m`
- **Date Range:** 2025-06-01 00:00:00 to 2026-05-31 23:59:00
- **Scale Status:** Full 12-Month Dataset

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
