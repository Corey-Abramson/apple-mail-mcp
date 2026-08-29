"""Account name ↔ UUID mapping for FTS5 search filters.

The disk indexer stores account UUIDs from filesystem paths
(e.g., "24E569DF-5E45-...") but users pass friendly names
(e.g., "Work") from JXA. This module bridges the gap.

Usage:
    acct_map = AccountMap.get_instance()
    await acct_map.ensure_loaded()
    uuid = acct_map.name_to_uuid("Work")   # → "24E569DF-..."
    name = acct_map.uuid_to_name("24E5...")  # → "Work"
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Cache TTL in seconds (5 minutes)
_CACHE_TTL = 300


class AccountMap:
    """Thread-safe, cached mapping between account names and UUIDs.

    JXA `Mail.accounts.id()` returns the same UUIDs as the
    filesystem folder names under ~/Library/Mail/V10/, so we
    can reliably translate friendly names for FTS5 queries.
    """

    _instance: AccountMap | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._name_to_uuid: dict[str, str] = {}
        self._uuid_to_name: dict[str, str] = {}
        self._ambiguous_names: set[str] = set()
        self._source: str | None = None
        self._loaded_at: float = 0
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> AccountMap:
        """Get the singleton AccountMap instance (thread-safe)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = AccountMap()
            return cls._instance

    def name_to_uuid(self, name: str) -> str | None:
        """Translate a friendly account name to its UUID.

        Args:
            name: Friendly account name (e.g., "Work")

        Returns:
            UUID string, or None if not found
        """
        with self._lock:
            return self._name_to_uuid.get(name)

    def names_to_uuids(self, names: set[str]) -> set[str]:
        """Resolve a set of account names to their UUIDs.

        Exact, case-sensitive lookups against the loaded map. Names
        with no matching account are dropped and logged at WARNING —
        important for the exclusion feature, where a typo'd name
        would otherwise silently fail to hide an account. The map
        must already be loaded (``ensure_loaded`` / ``load_from_jxa``).
        """
        with self._lock:
            resolved = {
                self._name_to_uuid[n] for n in names if n in self._name_to_uuid
            }
            unknown = {n for n in names if n not in self._name_to_uuid}
            source = self._source or "unloaded"
        if unknown:
            logger.warning(
                "Account exclusion: no account named %s; not hidden. "
                "Names are exact and case-sensitive (map source: %s).",
                ", ".join(sorted(repr(n) for n in unknown)),
                source,
            )
        return resolved

    def reset(self) -> None:
        """Forget all cached state. For tests and explicit refresh."""
        with self._lock:
            self._name_to_uuid.clear()
            self._uuid_to_name.clear()
            self._ambiguous_names.clear()
            self._source = None
            self._loaded_at = 0

    def get_cached_accounts(self) -> list[dict[str, str]] | None:
        """Return the cached account list, or None if cold/stale.

        Lets `list_accounts()` serve repeat calls without a JXA
        round-trip. Returns `[{"name": ..., "id": ...}, ...]`
        when the cache is fresh, else None — caller falls back
        to JXA to populate.
        """
        with self._lock:
            if self._loaded_at == 0:
                return None
            if (time.monotonic() - self._loaded_at) > _CACHE_TTL:
                return None
            return [
                {"name": name, "id": uid}
                for uid, name in self._uuid_to_name.items()
            ]

    def uuid_to_name(self, uuid: str) -> str:
        """Translate a UUID to its friendly account name.

        Args:
            uuid: Account UUID from the index DB

        Returns:
            Friendly name, or the UUID itself as fallback
        """
        with self._lock:
            return self._uuid_to_name.get(uuid, uuid)

    def _load_accounts(self, accounts: list[dict], *, source: str) -> int:
        """Populate both directions, refusing ambiguous name lookups."""
        name_to_uuids: dict[str, set[str]] = {}
        uuid_to_name: dict[str, str] = {}
        for acct in accounts:
            name = acct.get("name", "")
            uid = acct.get("id", "")
            if name and uid:
                name_to_uuids.setdefault(name, set()).add(uid)
                uuid_to_name[uid] = name

        ambiguous = {
            name for name, uuids in name_to_uuids.items() if len(uuids) > 1
        }
        name_to_uuid = {
            name: next(iter(uuids))
            for name, uuids in name_to_uuids.items()
            if name not in ambiguous
        }

        with self._lock:
            self._name_to_uuid = name_to_uuid
            self._uuid_to_name = uuid_to_name
            self._ambiguous_names = ambiguous
            self._source = source
            self._loaded_at = time.monotonic()

        if ambiguous:
            logger.warning(
                "AccountMap refused ambiguous display-name lookups for %s "
                "(map source: %s)",
                ", ".join(sorted(repr(name) for name in ambiguous)),
                source,
            )
        logger.debug(
            "AccountMap loaded: %d accounts (source: %s)",
            len(uuid_to_name),
            source,
        )
        return len(uuid_to_name)

    def load_from_jxa(self, accounts: list[dict]) -> None:
        """Populate from JXA output and persist the validated rows.

        Args:
            accounts: List of {"name": "Work", "id": "UUID"} dicts
        """
        self._load_accounts(accounts, source="jxa")
        try:
            from .manager import IndexManager

            IndexManager.get_instance().save_account_map(accounts)
        except Exception:
            logger.warning(
                "Could not persist the JXA account display-name map",
                exc_info=True,
            )

    def load_from_db(self) -> bool:
        """Load the persisted map without contacting Mail.app."""
        try:
            from .manager import IndexManager

            accounts = IndexManager.get_instance().load_account_map()
        except Exception:
            logger.debug(
                "Could not read the persisted account display-name map",
                exc_info=True,
            )
            return False
        if not accounts:
            return False
        self._load_accounts(accounts, source="db")
        return True

    def _is_stale(self) -> bool:
        """Check if the cache needs refreshing."""
        if self._loaded_at == 0:
            return True
        return (time.monotonic() - self._loaded_at) > _CACHE_TTL

    async def ensure_loaded(self, *, allow_jxa: bool = True) -> None:
        """Ensure the map is populated, fetching via JXA if needed.

        Called from async context (MCP tool handlers). Uses
        execute_with_core_async to avoid blocking the event loop.

        Double-checked locking prevents concurrent callers from
        firing duplicate JXA fetches.
        """
        if not self._is_stale():
            return

        async with self._async_lock:
            # Re-check after acquiring lock — another coroutine
            # may have refreshed while we were waiting.
            if not self._is_stale():
                return

            if self.load_from_db():
                return

            if not allow_jxa:
                return

            from ..builders import AccountsQueryBuilder
            from ..executor import execute_with_core_async

            script = AccountsQueryBuilder().list_accounts()
            accounts = await execute_with_core_async(script)
            self.load_from_jxa(accounts)


def resolve_excluded_account_uuids(
    names: set[str] | None = None,
) -> set[str]:
    """Resolve excluded account names to UUIDs from a synchronous context.

    The index build, disk sync, watcher, and CLI first use the persisted
    account map. JXA is only a bootstrap fallback when that table is
    empty and exclusions actually require name resolution.

    ``names=None`` (the default) reads the configured exclusions via
    :func:`~apple_mail_mcp.config.get_index_exclude_accounts`, so the
    index/sync/watcher callers can never diverge on which config key
    feeds the exclusion set.

    Returns an empty set (and logs) if neither source can resolve the
    map. Server read paths treat that total failure as fail-closed when
    exclusions are configured.
    """
    if names is None:
        from ..config import get_index_exclude_accounts

        names = get_index_exclude_accounts()
    if not names:
        return set()

    acct_map = AccountMap.get_instance()
    if acct_map.get_cached_accounts() is None:  # cold or stale
        if not acct_map.load_from_db():
            from ..builders import AccountsQueryBuilder
            from ..executor import execute_with_core

            try:
                accounts = execute_with_core(
                    AccountsQueryBuilder().list_accounts()
                )
            except Exception as exc:
                logger.warning(
                    "Could not resolve excluded accounts from the "
                    "persisted map or JXA (%s); no accounts excluded "
                    "at the index layer this run.",
                    exc,
                )
                return set()
            acct_map.load_from_jxa(accounts)
    return acct_map.names_to_uuids(names)
