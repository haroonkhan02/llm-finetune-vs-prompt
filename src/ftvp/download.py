"""Download MASSIVE 1.1 and extract one locale (plus the licence) from the archive.

The archive is ~40 MB with 51 locales; only ``<locale>.jsonl`` is kept. The download
resumes a partial file, is checked against a pinned SHA-256, and archive members are
matched by exact name, so nothing outside ``raw_dir`` can be written.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from ftvp.config import load_config, resolve

ARCHIVE_NAME = "massive-1.1.tar.gz"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest``, resuming from a partial ``.part`` file if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if start and resp.status == 206 else "wb"
        with open(part, mode) as out:
            shutil.copyfileobj(resp, out, 1 << 20)
    part.replace(dest)
    return dest


def extract_members(archive: Path, wanted: dict[str, Path]) -> list[Path]:
    """Extract exactly the archive members named in ``wanted`` (member -> target path)."""
    written = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = wanted.get(member.name)
            if target is None or not member.isfile():
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written.append(target)
    missing = set(wanted.values()) - set(written)
    if missing:
        raise FileNotFoundError(f"not found in {archive.name}: {sorted(map(str, missing))}")
    return written


def locale_file(raw_dir: Path, locale: str) -> Path:
    return Path(raw_dir) / f"{locale}.jsonl"


def download(url: str, sha256: str, raw_dir: Path, locale: str) -> Path:
    raw_dir = Path(raw_dir)
    out = locale_file(raw_dir, locale)
    if out.exists():
        return out
    archive = raw_dir / ARCHIVE_NAME
    if not archive.exists():
        fetch(url, archive)
    digest = sha256_of(archive)
    if digest != sha256:
        archive.unlink()
        raise ValueError(f"checksum mismatch for {ARCHIVE_NAME}: {digest}; deleted, rerun")
    extract_members(archive, {f"1.1/data/{locale}.jsonl": out,
                              "1.1/LICENSE": raw_dir / "LICENSE"})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    ds = load_config(args.config)["dataset"]
    path = download(ds["url"], ds["sha256"], resolve(ds["raw_dir"]), ds["locale"])
    print(f"{path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
