from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyotp
from SmartApi import SmartConnect

from .config import get_provider_credentials

LOGGER = logging.getLogger(__name__)


class CandleDataMode(str, Enum):
    """SmartAPI candles are raw; adjusted bars require a local action file."""

    RAW = "raw"
    ADJUSTED = "adjusted"


class BrokerApiSource:
    """Small broker boundary around an authorized SmartAPI session."""

    def __init__(self, api: Any, *, min_request_interval: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        if min_request_interval < 0:
            raise ValueError("min_request_interval must be >= 0")
        self.api = api
        self.min_request_interval = min_request_interval
        self._clock = clock
        self._last_attempt_at: float | None = None

    def _throttle(self, sleep: Callable[[float], None]) -> None:
        if self._last_attempt_at is not None:
            wait = self.min_request_interval - (self._clock() - self._last_attempt_at)
            if wait > 0:
                sleep(wait)
        self._last_attempt_at = self._clock()

    def candles(self, params: dict[str, str], *, retries: int,
                sleep: Callable[[float], None]) -> dict[str, Any]:
        for attempt in range(retries):
            self._throttle(sleep)
            try:
                response = self.api.getCandleData(params)
                if not isinstance(response, dict) or response.get("status") is not True:
                    raise RuntimeError("unsuccessful candle response")
                return response
            except Exception:
                if attempt + 1 >= retries:
                    raise
                sleep(min(2 ** attempt, 8))
                self._last_attempt_at = self._clock()
        raise AssertionError("unreachable")

    def instrument_master(self) -> Any:
        return self.api.getScripMaster()


class InstrumentMasterSource:
    """Fetch and parse the SmartAPI instrument master using an injected client."""

    REQUIRED_COLUMNS = ("token", "symbol", "name", "expiry", "strike", "lotsize",
                        "instrumenttype", "exch_seg", "tick_size")

    def __init__(self, *, client: Any | None = None, dry_run: bool = False,
                 client_factory: Callable[[], Any] | None = None):
        self._client = client
        self._dry_run = dry_run
        self._client_factory = client_factory

    def fetch(self) -> pd.DataFrame:
        if self._dry_run:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        client = self._client
        if client is None:
            if self._client_factory is None:
                raise RuntimeError("instrument master requires an authenticated client")
            client = self._client_factory()
        payload = client.instrument_master() if isinstance(client, BrokerApiSource) else client.getScripMaster()
        try:
            records = json.loads(payload) if isinstance(payload, str) else payload
        except (TypeError, ValueError):
            raise RuntimeError("Angel One instrument master response is invalid") from None
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise RuntimeError("Angel One instrument master response is invalid")
        frame = pd.DataFrame(records)
        missing = set(self.REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise RuntimeError("Angel One instrument master schema is missing required fields")
        frame["token"] = frame["token"].astype(str)
        frame["exch_seg"] = frame["exch_seg"].astype(str)
        return frame


@dataclass(frozen=True)
class CandleRequest:
    exchange: str
    symbol_token: str
    interval: str
    from_date: str
    to_date: str


class AngelOneClient:
    """Login and fetch historical candles; caller supplies authorized symbol tokens."""

    def __init__(self, credentials: dict[str, str] | None = None, *, dry_run: bool = False,
                 min_request_interval: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        if min_request_interval < 0:
            raise ValueError("min_request_interval must be >= 0")
        self._dry_run = dry_run
        self._credentials = {} if dry_run else (credentials if credentials is not None else get_provider_credentials())
        self._min_request_interval = min_request_interval
        self._clock = clock
        required = ("ANGELONE_API_KEY", "ANGELONE_CLIENT_CODE", "ANGELONE_PASSWORD", "ANGELONE_TOTP_SECRET")
        missing = [] if dry_run else [name for name in required if not self._credentials.get(name)]
        if missing:
            raise RuntimeError("Missing Angel One credential settings: " + ", ".join(missing))
        self._api: SmartConnect | None = None
        self._broker: BrokerApiSource | None = None
        self._last_request_at: float | None = None

    def login(self):
        if self._dry_run:
            raise RuntimeError("login is disabled in dry-run mode")
        if self._api is not None:
            return
        api = SmartConnect(api_key=self._credentials["ANGELONE_API_KEY"])
        otp = pyotp.TOTP(self._credentials["ANGELONE_TOTP_SECRET"]).now()
        response = api.generateSession(
            self._credentials["ANGELONE_CLIENT_CODE"],
            self._credentials["ANGELONE_PASSWORD"],
            otp,
        )
        if not isinstance(response, dict) or response.get("status") is not True:
            try:
                api.terminateSession(self._credentials["ANGELONE_CLIENT_CODE"])
            except Exception:
                pass
            raise RuntimeError("Angel One login failed; check credentials, TOTP clock, and API app settings")
        self._api = api
        self._broker = BrokerApiSource(
            api, min_request_interval=self._min_request_interval, clock=self._clock
        )

    def fetch(self, request: CandleRequest, *, retries: int = 3,
              sleep: Callable[[float], None] = time.sleep,
              mode: CandleDataMode = CandleDataMode.RAW,
              corporate_action_file: str | Path | None = None) -> pd.DataFrame:
        validate_request(request)
        if retries < 1:
            raise ValueError("retries must be at least 1")
        if not isinstance(mode, CandleDataMode):
            raise ValueError("mode must be CandleDataMode.RAW or CandleDataMode.ADJUSTED")
        if self._dry_run:
            frame = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            return transform_candles(frame, mode, corporate_action_file=corporate_action_file)
        self.login()
        assert self._broker is not None
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._min_request_interval - (now - self._last_request_at)
            if wait > 0:
                sleep(wait)
        self._last_request_at = self._clock()
        params = {
            "exchange": request.exchange,
            "symboltoken": request.symbol_token,
            "interval": request.interval,
            "fromdate": request.from_date,
            "todate": request.to_date,
        }
        try:
            response = self._broker.candles(params, retries=retries, sleep=sleep)
        except Exception as exc:
            LOGGER.error("SmartAPI candle request failed (%s)", type(exc).__name__)
            raise RuntimeError("Angel One candle request failed; check API limits, token, date span, and session") from None
        rows = response.get("data") or []
        names = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        if not isinstance(rows, list) or any(not isinstance(row, (tuple, list)) for row in rows):
            raise RuntimeError("unexpected candle schema")
        if rows and (len(rows[0]) not in (6, 7) or any(len(row) != len(rows[0]) for row in rows)):
            raise RuntimeError("unexpected candle schema")
        frame = pd.DataFrame(rows, columns=names[:len(rows[0])] if rows else names[:6])
        if frame.empty:
            return transform_candles(frame, mode, corporate_action_file=corporate_action_file)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        for name in names[1:]:
            if name in frame:
                frame[name] = pd.to_numeric(frame[name], errors="raise")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        return transform_candles(frame, mode, corporate_action_file=corporate_action_file)

    def close(self):
        if self._api is not None:
            try:
                self._api.terminateSession(self._credentials["ANGELONE_CLIENT_CODE"])
            finally:
                self._api = None
                self._broker = None


def transform_candles(frame: pd.DataFrame, mode: CandleDataMode, *,
                      corporate_action_file: str | Path | None = None) -> pd.DataFrame:
    """Keep SmartAPI bars raw unless user supplies a split-action ledger.

    Ledger columns: effective_date, split_factor (new shares per old share).
    Historical OHLC before the action is divided by the cumulative split factor;
    volume is multiplied. This does not adjust cash dividends or other events.
    """
    if mode is CandleDataMode.RAW:
        return frame.copy()
    if mode is not CandleDataMode.ADJUSTED:
        raise ValueError("mode must be CandleDataMode.RAW or CandleDataMode.ADJUSTED")
    if corporate_action_file is None:
        raise ValueError("adjusted candles require an explicit corporate-action file")
    actions = pd.read_csv(corporate_action_file)
    if not {"effective_date", "split_factor"}.issubset(actions.columns):
        raise ValueError("corporate-action file requires effective_date and split_factor columns")
    if frame.empty:
        return frame.copy()
    effective = pd.to_datetime(actions["effective_date"], errors="raise")
    if effective.dt.tz is not None:
        effective = effective.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    effective = effective.dt.normalize()
    factors = pd.to_numeric(actions["split_factor"], errors="raise")
    if (factors <= 0).any() or factors.isna().any():
        raise ValueError("corporate-action split factors must be positive finite numbers")
    stamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert("Asia/Kolkata")
        local_dates = stamps.dt.tz_localize(None).dt.normalize()
    else:
        local_dates = stamps.dt.normalize()
    result = frame.copy()
    for column in ("open", "high", "low", "close", "volume"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    for action_date, factor in zip(effective, factors):
        before = local_dates <= action_date
        result.loc[before, ["open", "high", "low", "close"]] = result.loc[before, ["open", "high", "low", "close"]] / factor
        if "volume" in result:
            result.loc[before, "volume"] = result.loc[before, "volume"] * factor
    return result


def write_candles_parquet(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    symbol: str,
    symbol_token: str,
    request: CandleRequest,
    product_entitlement: str | None = None,
    adjustment_provenance: str | None = None,
    universe_provenance: str | None = None,
) -> Path:
    # Validate the data schema before provenance so malformed frames fail with
    # the actionable schema error even for callers using the legacy signature.
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"candle frame is missing columns: {sorted(required - set(frame.columns))}")
    validate_request(request)
    if not symbol.strip() or not re.fullmatch(r"[A-Za-z0-9_-]+", symbol_token):
        raise ValueError("symbol must be non-empty and symbol_token valid")
    if any(not isinstance(value, str) or not value.strip() for value in (
        product_entitlement, adjustment_provenance, universe_provenance
    )):
        raise ValueError("entitlement, adjustment, and universe provenance are required")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    manifest = path.with_suffix(path.suffix + ".manifest.txt")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(
        f"source=Angel One SmartAPI\nproduct_entitlement={product_entitlement}\n"
        f"symbol={symbol}\nsymbol_token={symbol_token}\n"
        f"exchange={request.exchange}\ninterval={request.interval}\n"
        f"from={request.from_date}\nto={request.to_date}\n"
        f"sha256={digest}\nschema={','.join(frame.columns)}\nparser_version=1\n"
        f"adjustment_provenance={adjustment_provenance}\nuniverse_provenance={universe_provenance}\n"
        f"retrieved_at_utc={pd.Timestamp.now(tz='UTC').isoformat()}\n"
        "permission_note=Fetcher requires user verification of current terms; do not redistribute raw data.\n",
        encoding="utf-8",
    )
    return path


def validate_request(request: CandleRequest) -> None:
    if request.exchange not in {"NSE", "NFO", "BSE", "MCX"}:
        raise ValueError("unsupported exchange")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", request.symbol_token):
        raise ValueError("invalid symbol token")
    start = datetime.strptime(request.from_date, "%Y-%m-%d %H:%M")
    end = datetime.strptime(request.to_date, "%Y-%m-%d %H:%M")
    if start > end:
        raise ValueError("from_date must be <= to_date")
