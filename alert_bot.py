"""
BTC/USDT Alert Bot — VWAP & POC (Volume Profile) Monitor
=========================================================
Runs 24/7 on a lightweight server (e.g. AWS EC2 t2.micro).
Reads price data from Binance via CCXT (no API key needed)
and sends Telegram alerts when the price approaches key levels.

Author : Senior Backend Developer
Stack  : Python 3.10+, ccxt, python-telegram-bot, pandas, pandas_ta
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

load_dotenv()

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOL: str = "BTC/USDT"
POLL_INTERVAL_SECONDS: int = 30          # How often we check the price
PROXIMITY_THRESHOLD_PCT: float = 0.001   # 0.1% proximity to trigger alert
COOLDOWN_SECONDS: int = 1800             # 30 minutes cooldown per level
VOLUME_PROFILE_BINS: int = 100           # Granularity for volume profile
OHLCV_TIMEFRAME: str = "1m"             # 1-minute candles for precision
OHLCV_LIMIT: int = 1440                 # 24 hours of 1-min candles
MAX_RECONNECT_WAIT: int = 300           # Max backoff wait in seconds (5 min)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("alert_bot")

# ──────────────────────────────────────────────
# Telegram Notifier
# ──────────────────────────────────────────────


class TelegramNotifier:
    """Thin async wrapper around python-telegram-bot."""

    def __init__(self, token: str, chat_id: str) -> None:
        if not token or not chat_id:
            raise ValueError(
                "TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set in .env"
            )
        self._bot = Bot(token=token)
        self._chat_id = chat_id

    async def send(self, text: str) -> None:
        """Send a message to the configured Telegram chat."""
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
            )
            logger.info("Telegram message sent successfully.")
        except TelegramError as exc:
            logger.error("Failed to send Telegram message: %s", exc)


# ──────────────────────────────────────────────
# Market Data Provider
# ──────────────────────────────────────────────


class MarketData:
    """Fetches OHLCV data from Binance using CCXT (read-only, no API key)."""

    def __init__(self) -> None:
        self._exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )

    def fetch_ohlcv(
        self,
        symbol: str = SYMBOL,
        timeframe: str = OHLCV_TIMEFRAME,
        limit: int = OHLCV_LIMIT,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: timestamp, open, high, low, close, volume."""
        raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def fetch_ticker_price(self, symbol: str = SYMBOL) -> float:
        """Return the last traded price."""
        ticker = self._exchange.fetch_ticker(symbol)
        return float(ticker["last"])


# ──────────────────────────────────────────────
# Indicator Calculations
# ──────────────────────────────────────────────


def compute_intraday_vwap(df: pd.DataFrame) -> float:
    """
    Calculate the intraday VWAP, resetting at 00:00 UTC each day.

    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    """
    now_utc = datetime.now(timezone.utc)
    start_of_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    # Filter candles that belong to today (UTC)
    intraday = df[df["timestamp"] >= start_of_day].copy()

    if intraday.empty or intraday["volume"].sum() == 0:
        # Fallback: use all available data if it's very early in the day
        intraday = df.copy()

    typical_price = (intraday["high"] + intraday["low"] + intraday["close"]) / 3
    vwap = (typical_price * intraday["volume"]).sum() / intraday["volume"].sum()
    return float(vwap)


def compute_poc(df: pd.DataFrame, bins: int = VOLUME_PROFILE_BINS) -> float:
    """
    Compute the Point of Control (POC) from a 24-hour Anchored Volume Profile.

    The POC is the price level where the most volume has been traded.
    We discretize the price range into `bins` buckets, accumulate volume
    per bucket, and return the midpoint of the bucket with the highest volume.
    """
    if df.empty:
        return 0.0

    price_low = df["low"].min()
    price_high = df["high"].max()

    if price_high == price_low:
        return float(price_low)

    # Create evenly-spaced price bins
    bin_edges = np.linspace(price_low, price_high, bins + 1)
    volume_per_bin = np.zeros(bins, dtype=np.float64)

    # Distribute each candle's volume across the bins its range spans
    for _, row in df.iterrows():
        candle_low = row["low"]
        candle_high = row["high"]
        candle_vol = row["volume"]

        # Find which bins this candle overlaps
        low_idx = max(0, np.searchsorted(bin_edges, candle_low, side="right") - 1)
        high_idx = min(bins - 1, np.searchsorted(bin_edges, candle_high, side="left"))

        if low_idx > high_idx:
            low_idx = high_idx

        # Spread volume evenly across overlapping bins
        span = high_idx - low_idx + 1
        if span > 0:
            volume_per_bin[low_idx : high_idx + 1] += candle_vol / span

    poc_bin_idx = int(np.argmax(volume_per_bin))
    poc_price = (bin_edges[poc_bin_idx] + bin_edges[poc_bin_idx + 1]) / 2.0
    return float(poc_price)


# ──────────────────────────────────────────────
# Cooldown Manager
# ──────────────────────────────────────────────


class CooldownManager:
    """
    Prevents alert spam by enforcing a cooldown period per level type.
    Once an alert fires for 'vwap' or 'poc', the same alert won't fire
    again until COOLDOWN_SECONDS have elapsed.
    """

    def __init__(self, cooldown_seconds: int = COOLDOWN_SECONDS) -> None:
        self._cooldown = cooldown_seconds
        self._last_alert: dict[str, float] = {}

    def can_alert(self, level_name: str) -> bool:
        last = self._last_alert.get(level_name, 0.0)
        return (time.time() - last) >= self._cooldown

    def record_alert(self, level_name: str) -> None:
        self._last_alert[level_name] = time.time()


# ──────────────────────────────────────────────
# Alert Engine
# ──────────────────────────────────────────────


class AlertEngine:
    """Orchestrates market data fetching, indicator computation, and alerting."""

    def __init__(self) -> None:
        self._market = MarketData()
        self._notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        self._cooldown = CooldownManager()
        self._last_vwap: float = 0.0
        self._last_poc: float = 0.0

    @staticmethod
    def _is_near(price: float, level: float, threshold: float) -> bool:
        """Check if price is within `threshold` percent of level."""
        if level == 0:
            return False
        return abs(price - level) / level <= threshold

    def _build_alert_message(
        self,
        price: float,
        vwap: float,
        poc: float,
        triggered_levels: list[str],
    ) -> str:
        """Format a rich Telegram alert message."""
        levels_str = " / ".join(triggered_levels)
        return (
            f"⚠️ <b>BTC cerca de zona de interés</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Precio actual:</b>  <code>${price:,.2f}</code>\n"
            f"📈 <b>Nivel VWAP:</b>     <code>${vwap:,.2f}</code>\n"
            f"🐋 <b>Nivel POC (Ballenas):</b> <code>${poc:,.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔔 Nivel(es) detectado(s): <b>{levels_str}</b>\n"
            f"💡 <b>Sugerencia:</b> Preparar orden pendiente.\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    async def tick(self) -> None:
        """Execute one monitoring cycle."""
        # 1. Fetch current price
        price = self._market.fetch_ticker_price()
        logger.info("BTC/USDT price: $%,.2f", price)

        # 2. Fetch 24h of 1-min OHLCV candles
        df = self._market.fetch_ohlcv()
        if df.empty:
            logger.warning("No OHLCV data returned. Skipping this cycle.")
            return

        # 3. Compute indicators
        vwap = compute_intraday_vwap(df)
        poc = compute_poc(df)
        self._last_vwap = vwap
        self._last_poc = poc
        logger.info("VWAP: $%,.2f | POC: $%,.2f", vwap, poc)

        # 4. Check proximity & send alerts
        triggered: list[str] = []

        if self._is_near(price, vwap, PROXIMITY_THRESHOLD_PCT):
            if self._cooldown.can_alert("vwap"):
                triggered.append("VWAP")
                self._cooldown.record_alert("vwap")
            else:
                logger.debug("VWAP alert on cooldown.")

        if self._is_near(price, poc, PROXIMITY_THRESHOLD_PCT):
            if self._cooldown.can_alert("poc"):
                triggered.append("POC")
                self._cooldown.record_alert("poc")
            else:
                logger.debug("POC alert on cooldown.")

        if triggered:
            message = self._build_alert_message(price, vwap, poc, triggered)
            await self._notifier.send(message)
        else:
            logger.info("No proximity alert triggered.")


# ──────────────────────────────────────────────
# Main Loop with Auto-Reconnect
# ──────────────────────────────────────────────


async def main() -> None:
    """Run the alert bot forever with exponential backoff on failures."""
    logger.info("=" * 50)
    logger.info("  BTC/USDT Alert Bot — Starting")
    logger.info("  Symbol: %s | Interval: %ds", SYMBOL, POLL_INTERVAL_SECONDS)
    logger.info("  Proximity threshold: %.2f%%", PROXIMITY_THRESHOLD_PCT * 100)
    logger.info("  Alert cooldown: %d minutes", COOLDOWN_SECONDS // 60)
    logger.info("=" * 50)

    engine = AlertEngine()
    consecutive_failures: int = 0

    while True:
        try:
            await engine.tick()
            consecutive_failures = 0  # Reset on success
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
            consecutive_failures += 1
            wait = min(
                2**consecutive_failures * 5, MAX_RECONNECT_WAIT
            )
            logger.warning(
                "Network error (%s). Retrying in %ds… (attempt #%d)",
                exc.__class__.__name__,
                wait,
                consecutive_failures,
            )
            await asyncio.sleep(wait)

        except ccxt.ExchangeError as exc:
            consecutive_failures += 1
            wait = min(2**consecutive_failures * 5, MAX_RECONNECT_WAIT)
            logger.error(
                "Exchange error: %s. Retrying in %ds…", exc, wait
            )
            await asyncio.sleep(wait)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully…")
            break

        except Exception as exc:
            consecutive_failures += 1
            wait = min(2**consecutive_failures * 10, MAX_RECONNECT_WAIT)
            logger.exception(
                "Unexpected error: %s. Retrying in %ds…", exc, wait
            )
            await asyncio.sleep(wait)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
