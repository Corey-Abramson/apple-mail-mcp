"""Tests for the cross-process index writer lock (#106).

flock is tied to the open file description, not the process, so two
IndexLock instances contend properly even inside a single pytest
process — winner/loser/promotion are all testable without spawning
subprocesses.
"""

import stat
import threading
import time
from unittest.mock import MagicMock

import pytest

from apple_mail_mcp.cli import _index_writer_retry_loop
from apple_mail_mcp.index.lock import IndexLock


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "index.db"


class TestIndexLock:
    def test_first_acquire_wins(self, db_path):
        lock = IndexLock(db_path)
        assert lock.try_acquire() is True
        assert lock.locked is True

    def test_second_instance_loses(self, db_path):
        winner = IndexLock(db_path)
        assert winner.try_acquire() is True

        loser = IndexLock(db_path)
        assert loser.try_acquire() is False
        assert loser.locked is False

    def test_try_acquire_is_idempotent_while_held(self, db_path):
        lock = IndexLock(db_path)
        assert lock.try_acquire() is True
        assert lock.try_acquire() is True

    def test_release_allows_reacquire(self, db_path):
        winner = IndexLock(db_path)
        assert winner.try_acquire() is True
        winner.release()
        assert winner.locked is False

        contender = IndexLock(db_path)
        assert contender.try_acquire() is True

    def test_release_when_not_held_is_safe(self, db_path):
        IndexLock(db_path).release()  # no raise

    def test_lockfile_sits_next_to_db(self, db_path):
        lock = IndexLock(db_path)
        assert lock.lock_path == db_path.parent / "index.db.lock"
        lock.try_acquire()
        assert lock.lock_path.exists()

    def test_lockfile_permissions(self, db_path):
        lock = IndexLock(db_path)
        lock.try_acquire()
        mode = stat.S_IMODE(lock.lock_path.stat().st_mode)
        assert mode == 0o600

    def test_acquire_times_out_while_held(self, db_path):
        winner = IndexLock(db_path)
        assert winner.try_acquire() is True

        loser = IndexLock(db_path)
        start = time.monotonic()
        assert loser.acquire(timeout=0.3) is False
        assert time.monotonic() - start >= 0.3

    def test_acquire_succeeds_once_released(self, db_path):
        winner = IndexLock(db_path)
        assert winner.try_acquire() is True

        loser = IndexLock(db_path)
        releaser = threading.Timer(0.3, winner.release)
        releaser.start()
        try:
            assert loser.acquire(timeout=5.0) is True
        finally:
            releaser.cancel()

    def test_context_manager_releases(self, db_path):
        lock = IndexLock(db_path)
        with lock:
            lock.try_acquire()
            assert lock.locked is True
        assert lock.locked is False
        assert IndexLock(db_path).try_acquire() is True


class TestIndexWriterRetryLoop:
    """The index-passive server's promotion loop (cli.py)."""

    def test_promotes_after_writer_releases(self, db_path):
        winner = IndexLock(db_path)
        assert winner.try_acquire() is True

        loser = IndexLock(db_path)
        assert loser.try_acquire() is False

        manager = MagicMock()
        manager.index_writer = False
        promoted = threading.Event()

        thread = threading.Thread(
            target=_index_writer_retry_loop,
            args=(loser, manager, promoted.set, 0.05),
            daemon=True,
        )
        thread.start()

        # Several retry ticks pass while the writer still holds it.
        time.sleep(0.2)
        assert not promoted.is_set()
        assert manager.index_writer is False

        winner.release()
        assert promoted.wait(timeout=2.0), "loser never promoted"
        thread.join(timeout=2.0)
        assert manager.index_writer is True
        assert loser.locked is True


class TestBlockedWriterReleasesLock:
    """A writer that cannot read the Mail store must not keep the lock.

    Claude Desktop spawns copies that may lack Full Disk Access. If
    one wins the writer lock and holds it for its lifetime, every copy
    that *can* read Mail stays index-passive and the index freezes.
    """

    def _serve(self, db_path, blocked, probe):
        from unittest.mock import patch

        import apple_mail_mcp.cli as cli

        cli._sync_blocked_warned = False
        synced = threading.Event()

        manager = MagicMock()
        manager.has_index.return_value = True
        manager.db_path = db_path
        manager.index_writer = True
        manager.last_sync_blocked = None

        def sync_updates():
            manager.last_sync_blocked = blocked
            synced.set()
            return 0

        manager.sync_updates.side_effect = sync_updates

        mcp = MagicMock()
        mcp.run.side_effect = lambda: probe(synced, manager)

        with (
            patch(
                "apple_mail_mcp.index.IndexManager.get_instance",
                return_value=manager,
            ),
            patch("apple_mail_mcp.server.mcp", mcp),
            patch("apple_mail_mcp.server._cleanup_old_attachments"),
            patch(
                "apple_mail_mcp.config.get_lock_retry_seconds",
                return_value=3600,
            ),
        ):
            cli._run_serve(watch=False)

    def _contender_wins(self, db_path, timeout=2.0):
        contender = IndexLock(db_path)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if contender.try_acquire():
                    return True
                time.sleep(0.02)
            return False
        finally:
            contender.release()

    def test_permission_blocked_writer_releases_lock(self, db_path):
        state = {}

        def probe(synced, manager):
            assert synced.wait(timeout=2.0)
            state["won"] = self._contender_wins(db_path)
            state["index_writer"] = manager.index_writer

        self._serve(db_path, "permission", probe)
        assert state["won"] is True, "blocked writer kept the lock"
        assert state["index_writer"] is False

    def test_healthy_writer_keeps_lock(self, db_path):
        state = {}

        def probe(synced, manager):
            assert synced.wait(timeout=2.0)
            state["won"] = self._contender_wins(db_path, timeout=0.3)
            state["index_writer"] = manager.index_writer

        self._serve(db_path, None, probe)
        assert state["won"] is False
        assert state["index_writer"] is True
