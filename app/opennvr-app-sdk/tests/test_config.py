# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""Generic YAML config helper tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from opennvr_app_sdk.config import load_yaml, require


def test_load_yaml_returns_mapping(tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("nats_url: nats://nats:4222\nthreshold_seconds: 60\n")
    raw = load_yaml(cfg)
    assert raw == {"nats_url": "nats://nats:4222", "threshold_seconds": 60}


def test_load_yaml_rejects_non_mapping_root(tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml(cfg)


def test_load_yaml_rejects_empty_file(tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml(cfg)


def test_load_yaml_missing_file_raises_oserror(tmp_path: Path):
    with pytest.raises(OSError):
        load_yaml(tmp_path / "nope.yml")


def test_require_returns_value():
    assert require({"k": "v"}, "k") == "v"
    assert require({"n": 0.5}, "n") == 0.5


def test_require_rejects_missing_and_empty():
    with pytest.raises(ValueError, match="'k' is required"):
        require({}, "k")
    with pytest.raises(ValueError, match="'k' is required"):
        require({"k": None}, "k")
    with pytest.raises(ValueError, match="'k' is required"):
        require({"k": "   "}, "k")


def test_require_error_names_the_config_path():
    with pytest.raises(ValueError, match=r"config\.yml: 'nats_url' is required"):
        require({}, "nats_url", path="config.yml")
