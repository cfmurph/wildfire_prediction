#!/usr/bin/env python3
"""
Backup
=======
Copy what cannot be recreated to another location: the database, the input archive and the models.
ECMWF deletes its forecasts after about 4 days, so a lost archive is gone for good.

Nothing is deleted at the destination except database dumps beyond the newest KEEP_DUMPS.

Usage:
  python -m scripts.backup   (from the repo root; needs BACKUP_DIR and DATABASE_URL, api/.env)

The daily spread run calls this too.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

ARCHIVE_DIR = Path("data/live/archive")
MODEL_DIR = Path("experiments/checkpoints/spread")
KEEP_DUMPS = 14
# launchd's PATH does not include Homebrew's Postgres
PG_BIN = "/opt/homebrew/opt/postgresql@17/bin"


def _tool(name: str) -> str:
    return shutil.which(name, path=f"{os.environ.get('PATH', '')}:{PG_BIN}") or name


def backup(dest: Path, database_url: str) -> None:
    """Dump the database, checking the dump reads back, then copy the archive and models across."""
    if not dest.is_dir():  # an unplugged disk: creating the path would fill this disk instead
        raise FileNotFoundError(f"Backup destination {dest} is not available")
    dumps = dest / "db"
    dumps.mkdir(exist_ok=True)
    tmp = dumps / "tmp_wildfire.dump"
    try:
        subprocess.run([_tool("pg_dump"), "--format=custom", f"--file={tmp}", database_url],
                       check=True, capture_output=True)
        subprocess.run([_tool("pg_restore"), "--list", str(tmp)], check=True, capture_output=True)
        tmp.replace(dumps / f"wildfire_{date.today():%Y-%m-%d}.dump")
    finally:
        tmp.unlink(missing_ok=True)
    for old in sorted(dumps.glob("wildfire_*.dump"))[:-KEEP_DUMPS]:
        old.unlink()
    for source, name in [(ARCHIVE_DIR, "archive"), (MODEL_DIR, "models")]:
        subprocess.run(["rsync", "-a", "--exclude=tmp_*", f"{source}/", f"{dest / name}/"],
                       check=True, capture_output=True)
    log.info(f"Backed up to {dest}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    load_dotenv("api/.env")
    backup(Path(os.environ["BACKUP_DIR"]).expanduser(), os.environ["DATABASE_URL"])
