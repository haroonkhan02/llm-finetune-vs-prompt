import pytest

from ftvp.config import REPO_ROOT, load_config, resolve


def test_default_config_loads():
    cfg = load_config()
    assert cfg["dataset"]["locale"] == "en-US"
    assert cfg["eval"]["test_subset"] > 0


def test_missing_section_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset: {}\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_resolve_relative_to_repo():
    assert resolve("data/raw") == REPO_ROOT / "data" / "raw"
