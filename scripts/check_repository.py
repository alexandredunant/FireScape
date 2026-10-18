#!/usr/bin/env python3
"""Run lightweight integrity checks for the publication repository."""

from __future__ import annotations

import re
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.tex"
MAX_FILE_BYTES = 50 * 1024 * 1024


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "CITATION.bib",
        ROOT / "LICENSE",
        ROOT / "environment.yml",
        ROOT / "report" / "FireScape_forestry_report.pdf",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        fail(f"missing required files: {', '.join(missing)}")

    report_text = REPORT.read_text(encoding="utf-8")
    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", report_text)
    missing_graphics = [name for name in graphics if not (REPORT.parent / name).is_file()]
    if missing_graphics:
        fail(f"missing report figures: {', '.join(missing_graphics)}")

    for source in (ROOT / "src").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        if "/mnt/CEPH_PROJECTS/" in text:
            fail(f"machine-specific path remains in {source.relative_to(ROOT)}")

    oversized = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.stat().st_size > MAX_FILE_BYTES
    ]
    if oversized:
        fail(f"files larger than 50 MiB: {', '.join(map(str, oversized))}")

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for field in ("cff-version:", "title:", "authors:", "version:"):
        if field not in cff:
            fail(f"CITATION.cff lacks {field}")

    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"checksum mismatch for {relative}")

    print(f"OK: {len(graphics)} report figures resolved; checksums and integrity checks passed")


if __name__ == "__main__":
    main()
