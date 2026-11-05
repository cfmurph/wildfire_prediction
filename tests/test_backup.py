"""The backup's safety rules; its commands are faked, so no database or rsync is needed."""

import subprocess

import pytest

from scripts import backup as backup_module
from scripts.backup import KEEP_DUMPS, backup


@pytest.fixture
def commands(monkeypatch):
    """Records each command; pg_dump writes its file as the real one does."""
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "pg_dump" in command[0]:
            with open(next(a for a in command if a.startswith("--file=")).removeprefix("--file="), "w") as f:
                f.write("dump")

    monkeypatch.setattr(backup_module.subprocess, "run", run)
    return calls


def test_backup_dumps_checks_then_copies_without_deleting(commands, tmp_path):
    backup(tmp_path, "postgresql://localhost/wildfire")
    tools = [c[0].rsplit("/", 1)[-1] for c in commands]
    assert tools == ["pg_dump", "pg_restore", "rsync", "rsync"]
    assert all("--delete" not in " ".join(c) for c in commands)
    assert commands[2][-1] == f"{tmp_path / 'archive'}/"
    assert commands[3][-1] == f"{tmp_path / 'models'}/"
    dumps = list((tmp_path / "db").iterdir())
    assert len(dumps) == 1 and dumps[0].name.startswith("wildfire_")


def test_a_missing_destination_is_an_error_not_a_new_folder(commands, tmp_path):
    missing = tmp_path / "unplugged disk"
    with pytest.raises(FileNotFoundError, match="not available"):
        backup(missing, "postgresql://localhost/wildfire")
    assert not missing.exists()
    assert commands == []


def test_only_the_newest_dumps_are_kept(commands, tmp_path):
    (tmp_path / "db").mkdir()
    for day in range(1, KEEP_DUMPS + 5):
        (tmp_path / "db" / f"wildfire_2020-01-{day:02d}.dump").touch()
    (tmp_path / "db" / "wildfire_notes.txt").touch()
    backup(tmp_path, "postgresql://localhost/wildfire")
    dumps = sorted(p.name for p in (tmp_path / "db").glob("wildfire_*.dump"))
    assert len(dumps) == KEEP_DUMPS
    assert "wildfire_2020-01-01.dump" not in dumps
    assert (tmp_path / "db" / "wildfire_notes.txt").exists()


def test_a_dump_that_does_not_read_back_replaces_nothing(monkeypatch, tmp_path):
    (tmp_path / "db").mkdir()
    good = tmp_path / "db" / "wildfire_2020-01-01.dump"
    good.write_text("good")

    def run(command, **kwargs):
        if "pg_dump" in command[0]:
            (tmp_path / "db" / "tmp_wildfire.dump").write_text("corrupt")
        elif "pg_restore" in command[0]:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(backup_module.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        backup(tmp_path, "postgresql://localhost/wildfire")
    assert [p.name for p in (tmp_path / "db").iterdir()] == [good.name]
    assert good.read_text() == "good"
