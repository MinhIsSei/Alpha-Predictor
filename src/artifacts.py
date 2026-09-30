"""Keep old data snapshots and models instead of overwriting them.

ingest.py, features.py and train.py write to fixed "current" paths that the
rest of the pipeline reads (see config.py). Rather than give every output a
unique name — which would force every consumer to find "the latest" file —
the current path stays stable and whatever was there before is moved into an
`archive/` folder next to it first. Consumers are unchanged; nothing is lost.
"""

import hashlib
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def archive_existing(path: Path) -> Path | None:
    """Move `path` to `<dir>/archive/<stem>_<timestamp><suffix>` if it exists.

    The timestamp is the file's own last-modified time (UTC), i.e. when that
    snapshot was written, not when it was archived. Returns the archive path,
    or None if there was nothing to archive.
    """
    path = Path(path)
    if not path.exists():
        return None

    written_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{path.stem}_{written_at:%Y%m%dT%H%M%SZ}"
    destination = archive_dir / f"{stem}{path.suffix}"
    counter = 1
    while destination.exists():
        destination = archive_dir / f"{stem}_{counter}{path.suffix}"
        counter += 1

    shutil.move(str(path), destination)
    return destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_dir: Path) -> str | None:
    """Current commit hash, with "-dirty" if there are uncommitted changes.

    None if git isn't available or `repo_dir` isn't a repository — recording
    provenance is best-effort and must never stop training.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit
