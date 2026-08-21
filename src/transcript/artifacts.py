"""Intermediate artifact handling: atomic JSON writes, cached-stage skipping."""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

SCHEMA_VERSION = 1

log = logging.getLogger(__name__)


def utc_timestamp() -> str:
    """Now, as the ISO-8601 form artifacts stamp themselves with:
    2026-08-21T14:03:57Z."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


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
    """Load a stage artifact, rejecting anything the later stages cannot
    read. A truncated or hand-edited artifact is reported as such — the
    fix is always to re-run its stage with --force."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"artifact not found: {path}. Run the stage that produces it first."
        ) from exc
    except json.JSONDecodeError as exc:
        detail = "it is empty" if path.stat().st_size == 0 else f"{exc}"
        raise ValueError(
            f"{path} is not valid JSON ({detail}). An interrupted or failed "
            f"run can leave it incomplete; re-run the stage with --force."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not UTF-8 text ({exc}); re-run the stage with --force."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: expected a JSON object, found {type(payload).__name__}; "
            f"re-run the stage with --force."
        )
    version = payload.get("schema_version")
    if version != expected_schema:
        raise ValueError(
            f"{path}: schema_version {version!r} does not match expected "
            f"{expected_schema}; re-run the stage with --force"
        )
    return payload


def require_list(payload: dict, key: str, path: Path) -> list:
    """Read a list field an artifact is required to carry."""
    value = payload.get(key)
    if not isinstance(value, list):
        found = "missing" if value is None else f"a {type(value).__name__}"
        raise ValueError(
            f"{path}: expected a {key!r} list, found {found}; the file is not "
            f"an artifact of the expected stage — re-run it with --force."
        )
    return value


def should_skip(output: Path, force: bool) -> bool:
    """True if the stage output already exists and --force was not given."""
    output = Path(output)
    if force or not output.exists():
        return False
    log.info("Using cached %s (pass --force to regenerate)", output)
    return True
