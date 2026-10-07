import json
from pathlib import Path

import pytest

from pairs_trading.frozen_study import DEFAULT_CONFIG, main, validate_config


def test_checked_in_frozen_config_refuses_missing_local_dataset(capsys):
    assert main(["--config", str(DEFAULT_CONFIG)]) == 2
    assert "user-provided licensed dataset is missing" in capsys.readouterr().err


def test_valid_local_dataset_requires_rights_manifest_and_writes_only_input_status(tmp_path):
    source = tmp_path / "authorized.csv"
    source.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"
        "2016-01-01,A,10,11,9,10,100\n"
        "2024-12-31,A,20,21,19,20,200\n",
        encoding="utf-8",
    )
    import hashlib
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "provider": "authorized-local-export",
        "product_entitlement": "documented local research license",
        "rights_status": "authorized",
        "retrieved_at_utc": "2025-01-01T00:00:00+00:00",
        "covered_start": "2016-01-01",
        "covered_end": "2024-12-31",
        "parser_version": "test-1",
        "adjustment_provenance": "documented adjusted OHLCV",
        "universe_provenance": "historical security panel includes delisted securities where available",
        "sha256": checksum,
    }
    manifest_path = tmp_path / "authorized.csv.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"] = {"path": source.name, "manifest": manifest_path.name}
    config["outputs"]["run_manifest"] = "run.json"
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "inputs_validated_not_run"
    assert run["results"] is None


def test_rejects_unprovenanced_or_checksum_mismatched_dataset(tmp_path):
    source = tmp_path / "bars.csv"
    source.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"
        "2016-01-01,A,10,11,9,10,100\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "bars.csv.manifest.json"
    manifest_path.write_text(json.dumps({"provider": "unknown"}), encoding="utf-8")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"] = {"path": source.name, "manifest": manifest_path.name}
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.provider"):
        validate_config(config, config_path)
