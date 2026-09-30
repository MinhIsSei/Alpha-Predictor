import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from src.artifacts import archive_existing, file_sha256, git_commit


@contextmanager
def temp_dir():
    path = tempfile.mkdtemp()
    try:
        yield Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestArchiveExisting(unittest.TestCase):
    def test_missing_file_returns_none(self):
        with temp_dir() as tmp:
            self.assertIsNone(archive_existing(tmp / "model.joblib"))
            self.assertFalse((tmp / "archive").exists())

    def test_moves_file_into_archive_named_by_its_write_time(self):
        with temp_dir() as tmp:
            current = tmp / "model.joblib"
            current.write_bytes(b"old model")
            os.utime(current, (1_789_000_000, 1_789_000_000))  # 2026-09-10T00:26:40Z

            archived = archive_existing(current)

            self.assertFalse(current.exists())
            self.assertEqual(archived, tmp / "archive" / "model_20260910T002640Z.joblib")
            self.assertEqual(archived.read_bytes(), b"old model")

    def test_same_timestamp_does_not_overwrite_an_earlier_archive(self):
        with temp_dir() as tmp:
            current = tmp / "data.parquet"
            archived = []
            for content in (b"first", b"second"):
                current.write_bytes(content)
                os.utime(current, (1_789_000_000, 1_789_000_000))
                archived.append(archive_existing(current))

            self.assertNotEqual(archived[0], archived[1])
            self.assertEqual(archived[0].read_bytes(), b"first")
            self.assertEqual(archived[1].read_bytes(), b"second")


class TestProvenance(unittest.TestCase):
    def test_sha256_matches_known_digest(self):
        with temp_dir() as tmp:
            path = tmp / "f.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                file_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_git_commit_is_none_outside_a_repository(self):
        with temp_dir() as tmp:
            self.assertIsNone(git_commit(tmp))


if __name__ == "__main__":
    unittest.main()
