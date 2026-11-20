import io
import tarfile

import pytest

from ftvp.download import download, extract_members, sha256_of


def _make_archive(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_only_named_members(tmp_path):
    arc = tmp_path / "a.tar.gz"
    _make_archive(arc, {"1.1/data/en-US.jsonl": b"{}\n", "1.1/data/fr-FR.jsonl": b"x",
                        "../evil.txt": b"nope", "1.1/LICENSE": b"CC BY 4.0"})
    out = tmp_path / "raw"
    written = extract_members(arc, {"1.1/data/en-US.jsonl": out / "en-US.jsonl",
                                    "1.1/LICENSE": out / "LICENSE"})
    assert sorted(p.name for p in written) == ["LICENSE", "en-US.jsonl"]
    assert not (tmp_path / "evil.txt").exists()
    assert not (out / "fr-FR.jsonl").exists()


def test_missing_member_raises(tmp_path):
    arc = tmp_path / "a.tar.gz"
    _make_archive(arc, {"1.1/LICENSE": b"x"})
    with pytest.raises(FileNotFoundError):
        extract_members(arc, {"1.1/data/en-US.jsonl": tmp_path / "en-US.jsonl"})


def test_download_uses_cached_archive_and_checks_hash(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    arc = raw / "massive-1.1.tar.gz"
    _make_archive(arc, {"1.1/data/en-US.jsonl": b"{}\n", "1.1/LICENSE": b"x"})
    with pytest.raises(ValueError):
        download("http://unused", "0" * 64, raw, "en-US")
    assert not arc.exists()  # a bad archive is deleted so the next run re-fetches
    _make_archive(arc, {"1.1/data/en-US.jsonl": b"{}\n", "1.1/LICENSE": b"x"})
    path = download("http://unused", sha256_of(arc), raw, "en-US")
    assert path.read_bytes() == b"{}\n"
