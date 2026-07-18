"""Intermediate artifact handling: atomic JSON writes, cached-stage skipping."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

SCHEMA_VERSION = 1

log = logging.getLogger(__name__)


def save_json(path: Path, payload: dict) -> None:
    """Write JSON atomically (tmp + rename) so an interrupted run never
    leaves a corrupt file that skip-if-exists logic would later trust."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def load_json(path: Path, expected_schema: int = SCHEMA_VERSION) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    version = payload.get("schema_version")
    if version != expected_schema:
        raise ValueError(
            f"{path}: schema_version {version!r} does not match expected "
            f"{expected_schema}; re-run the stage with --force"
        )
    return payload


def should_skip(output: Path, force: bool) -> bool:
    """True if the stage output already exists and --force was not given."""
    output = Path(output)
    if force or not output.exists():
        return False
    log.info("Using cached %s (pass --force to regenerate)", output)
    return True
