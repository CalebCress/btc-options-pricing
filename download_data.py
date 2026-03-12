"""Download 6 months of BTCUSDT trade data from Tardis.

Usage:
    python download_data.py --api-key YOUR_API_KEY

Downloads:
    - binance spot trades (BTCUSDT)
    - binance-futures perp trades (BTCUSDT)
    - binance-futures derivative_ticker (funding rate, OI, mark/index price)
    - binance-futures liquidations
"""

import argparse
import logging
from tardis_dev import datasets

logging.basicConfig(level=logging.DEBUG)

FROM_DATE = "2025-09-01"
TO_DATE = "2026-03-12"
DOWNLOAD_DIR = "./data"


def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Tardis API key")
    args = parser.parse_args()

    # Binance spot trades
    print("=== Downloading Binance spot trades ===")
    datasets.download(
        exchange="binance",
        data_types=["trades"],
        from_date=FROM_DATE,
        to_date=TO_DATE,
        symbols=["BTCUSDT"],
        api_key=args.api_key,
        download_dir=DOWNLOAD_DIR,
        get_filename=file_name_nested,
    )

    # Binance USDT-M futures: trades + derivative ticker + liquidations
    print("=== Downloading Binance futures trades, ticker, liquidations ===")
    datasets.download(
        exchange="binance-futures",
        data_types=["trades", "derivative_ticker", "liquidations"],
        from_date=FROM_DATE,
        to_date=TO_DATE,
        symbols=["BTCUSDT"],
        api_key=args.api_key,
        download_dir=DOWNLOAD_DIR,
        get_filename=file_name_nested,
    )

    print("Done. Data saved to ./data/")


if __name__ == "__main__":
    main()
