from __future__ import annotations

import pandas as pd
import pytest

from pairs_trading.angelone import AngelOneClient, CandleRequest, write_candles_parquet, validate_request


class FakeSmartAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.login_args = None
        self.requests = []
        self.closed = False

    def generateSession(self, client_code, password, totp):
        self.login_args = (client_code, password, totp)
        return {"status": True}

    def getCandleData(self, params):
        self.requests.append(params)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def terminateSession(self, client_code):
        self.closed = True
        return {"status": True}


def credentials():
    return {
        "ANGELONE_API_KEY": "test-api-key",
        "ANGELONE_CLIENT_CODE": "test-client",
        "ANGELONE_PASSWORD": "test-pin",
        "ANGELONE_TOTP_SECRET": "JBSWY3DPEHPK3PXP",
    }


def test_dry_run_never_loads_credentials_constructs_api_or_calls_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run touched authentication/provider")

    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", forbidden)
    monkeypatch.setattr("pairs_trading.angelone.get_provider_credentials", forbidden)
    client = AngelOneClient(dry_run=True)
    result = client.fetch(
        CandleRequest("NSE", "123", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00")
    )
    assert result.empty
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_dry_run_prints_plan_and_does_not_write_artifacts(monkeypatch, capsys, tmp_path):
    from pairs_trading.fetch_angelone import main

    monkeypatch.setattr("sys.argv", ["pairs-fetch-angelone", "--symbol", "ABC", "--token", "123",
        "--from", "2024-01-01 00:00", "--to", "2024-01-02 00:00", "--out", str(tmp_path / "x.parquet"),
        "--dry-run", "--product-entitlement", "test", "--adjustment-provenance", "raw",
        "--universe-provenance", "test"])
    main()
    output = capsys.readouterr().out
    assert "dry run" in output.lower() and "ABC" in output and "123" in output
    assert not (tmp_path / "x.parquet").exists()


def test_request_rate_limiter_applies_between_provider_calls(monkeypatch):
    fake = FakeSmartAPI([
        {"status": True, "data": []},
        {"status": True, "data": []},
    ])
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: fake)
    monkeypatch.setattr("pairs_trading.angelone.pyotp.TOTP.now", lambda self: "123456")
    clock_values = iter([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
    pauses = []
    client = AngelOneClient(credentials(), min_request_interval=0.25, clock=lambda: next(clock_values))
    request = CandleRequest("NSE", "1", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00")
    client.fetch(request, sleep=pauses.append)
    client.fetch(request, sleep=pauses.append)
    assert pauses == [0.25]


def test_candle_data_mode_is_explicit_and_adjusted_mode_fails_closed():
    from pairs_trading.angelone import CandleDataMode, transform_candles

    assert CandleDataMode.RAW.value == "raw"
    assert CandleDataMode.ADJUSTED.value == "adjusted"
    with pytest.raises(ValueError, match="corporate-action file"):
        transform_candles(pd.DataFrame(), CandleDataMode.ADJUSTED)


def test_instrument_master_download_is_injectable_and_validated(monkeypatch):
    from pairs_trading.angelone import InstrumentMasterSource

    class FakeMaster:
        def __init__(self):
            self.calls = 0

        def getScripMaster(self):
            self.calls += 1
            return '[{"token":"123","symbol":"ABC-EQ","name":"ABC","expiry":"","strike":"-1","lotsize":"1","instrumenttype":"","exch_seg":"NSE","tick_size":"5"}]'

    fake = FakeMaster()
    source = InstrumentMasterSource(client=fake)
    frame = source.fetch()
    assert fake.calls == 1
    assert frame.iloc[0]["token"] == "123"
    assert frame.iloc[0]["exch_seg"] == "NSE"


def test_adjusted_output_requires_explicit_corporate_action_file(tmp_path):
    from pairs_trading.angelone import CandleDataMode, transform_candles

    raw = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"]),
        "open": [100.0], "high": [105.0], "low": [95.0], "close": [101.0], "volume": [10],
    })
    with pytest.raises(ValueError, match="corporate-action file"):
        transform_candles(raw, CandleDataMode.ADJUSTED)

    actions = tmp_path / "actions.csv"
    actions.write_text("effective_date,split_factor\n2024-01-01,2\n", encoding="utf-8")
    adjusted = transform_candles(raw, CandleDataMode.ADJUSTED, corporate_action_file=actions)
    assert adjusted.iloc[0].open == 50
    assert adjusted.iloc[0].high == 52.5
    assert adjusted.iloc[0].low == 47.5
    assert adjusted.iloc[0].close == 50.5
    assert adjusted.iloc[0].volume == 20


def test_instrument_master_dry_run_never_calls_provider():
    from pairs_trading.angelone import InstrumentMasterSource

    class Forbidden:
        def getScripMaster(self):
            raise AssertionError("dry run called instrument-master API")

    result = InstrumentMasterSource(client=Forbidden(), dry_run=True).fetch()
    assert result.empty
    assert "token" in result.columns


def test_adjusted_mode_in_client_uses_local_actions_only(monkeypatch, tmp_path):
    from pairs_trading.angelone import CandleDataMode

    fake = FakeSmartAPI([{"status": True, "data": [["2024-01-01T00:00:00+05:30", 100, 105, 95, 101, 10]]}])
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: fake)
    monkeypatch.setattr("pairs_trading.angelone.pyotp.TOTP.now", lambda self: "123456")
    actions = tmp_path / "actions.csv"
    actions.write_text("effective_date,split_factor\n2024-01-01,2\n", encoding="utf-8")
    client = AngelOneClient(credentials())
    result = client.fetch(
        CandleRequest("NSE", "1", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00"),
        mode=CandleDataMode.ADJUSTED,
        corporate_action_file=actions,
    )
    assert result.iloc[0].close == 50.5


def test_fetch_authenticates_maps_and_sorts_candles(monkeypatch):
    fake = FakeSmartAPI([{"status": True, "data": [
        ["2024-01-02T03:50:00+00:00", 12, 13, 11, 12.5, 100],
        ["2024-01-02T03:45:00+00:00", 11, 13, 10, 12, 90],
    ]}])
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: fake)
    monkeypatch.setattr("pairs_trading.angelone.pyotp.TOTP.now", lambda self: "123456")
    client = AngelOneClient(credentials())
    request = CandleRequest("NSE", "12345", "FIVE_MINUTE", "2024-01-02 09:15", "2024-01-02 15:30")
    frame = client.fetch(request)
    assert fake.login_args == ("test-client", "test-pin", "123456")
    assert frame.timestamp.is_monotonic_increasing
    assert frame.iloc[0].close == 12
    assert str(frame.timestamp.dt.tz) == "Asia/Kolkata"
    assert fake.requests == [{"exchange": "NSE", "symboltoken": "12345", "interval": "FIVE_MINUTE", "fromdate": "2024-01-02 09:15", "todate": "2024-01-02 15:30"}]


def test_fetch_retries_and_hides_provider_error(monkeypatch):
    fake = FakeSmartAPI([ValueError("contains private provider details"), {"status": True, "data": []}])
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: fake)
    monkeypatch.setattr("pairs_trading.angelone.pyotp.TOTP.now", lambda self: "123456")
    pauses = []
    client = AngelOneClient(credentials())
    frame = client.fetch(CandleRequest("NSE", "1", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00"), sleep=pauses.append)
    assert frame.empty
    assert pauses[0] == 1
    assert len(pauses) in (1, 2)
    assert len(fake.requests) == 2


def test_fetch_gives_sanitized_failure_after_retries(monkeypatch):
    fake = FakeSmartAPI([ValueError("secretish provider response")])
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: fake)
    monkeypatch.setattr("pairs_trading.angelone.pyotp.TOTP.now", lambda self: "123456")
    client = AngelOneClient(credentials())
    with pytest.raises(RuntimeError, match="check API limits") as exc:
        client.fetch(CandleRequest("NSE", "1", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00"), retries=1)
    assert "secretish" not in str(exc.value)


def test_close_terminates_session(monkeypatch):
    fake = FakeSmartAPI([])
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: fake)
    monkeypatch.setattr("pairs_trading.angelone.pyotp.TOTP.now", lambda self: "123456")
    client = AngelOneClient(credentials())
    client.login()
    client.close()
    assert fake.closed
    assert client._api is None


def test_invalid_candle_request_is_rejected_before_login(monkeypatch):
    constructed = []
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: constructed.append(api_key))
    client = AngelOneClient(credentials())
    with pytest.raises(ValueError, match="unsupported exchange"):
        client.fetch(CandleRequest("BAD", "1", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00"))
    assert constructed == []


def test_candle_request_rejects_malformed_token_without_normalizing():
    request = CandleRequest("NSE", " 00123", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00")
    with pytest.raises(ValueError, match="invalid symbol token"):
        validate_request(request)


def test_failed_login_never_saves_api_session(monkeypatch):
    fake = FakeSmartAPI([])
    fake.generateSession = lambda *args: {"status": False, "message": "private response"}
    monkeypatch.setattr("pairs_trading.angelone.SmartConnect", lambda api_key: fake)
    monkeypatch.setattr("pairs_trading.angelone.pyotp.TOTP.now", lambda self: "123456")
    client = AngelOneClient(credentials())
    with pytest.raises(RuntimeError, match="Angel One login failed") as exc:
        client.login()
    assert "private response" not in str(exc.value)
    assert client._api is None


def test_candle_writer_writes_checksum_and_provenance_manifest(tmp_path):
    output = tmp_path / "missing" / "candles.parquet"
    request = CandleRequest("NSE", "1", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00")
    empty = pd.DataFrame({"timestamp": pd.to_datetime([]), "open": [], "high": [], "low": [], "close": [], "volume": []})
    output = write_candles_parquet(
        empty, output, symbol="AAA", symbol_token="1", request=request,
        product_entitlement="test entitlement", adjustment_provenance="unadjusted", universe_provenance="fixture",
    )
    manifest = output.with_suffix(output.suffix + ".manifest.txt").read_text(encoding="utf-8")
    assert "sha256=" in manifest
    assert "product_entitlement=test entitlement" in manifest


def test_candle_writer_rejects_bad_schema_before_creating_files(tmp_path):
    output = tmp_path / "missing" / "candles.parquet"
    request = CandleRequest("NSE", "1", "ONE_DAY", "2024-01-01 00:00", "2024-01-02 00:00")
    with pytest.raises(ValueError, match="provenance"):
        write_candles_parquet(
            pd.DataFrame({"timestamp": pd.to_datetime([]), "open": [], "high": [], "low": [], "close": [], "volume": []}),
            output, symbol="AAA", symbol_token="1", request=request,
            product_entitlement="", adjustment_provenance="unadjusted", universe_provenance="fixture",
        )
    assert not output.parent.exists()
