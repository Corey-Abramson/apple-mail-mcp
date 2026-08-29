"""Tests for explicit account-map bootstrap."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apple_mail_mcp.cli import cli_sync_accounts


def test_sync_accounts_accepts_explicit_map(capsys):
    manager = MagicMock()
    manager.save_account_map.return_value = 2

    with patch(
        "apple_mail_mcp.index.IndexManager.get_instance",
        return_value=manager,
    ):
        cli_sync_accounts("Rice=UUID-1, Gmail=UUID-2")

    manager.save_account_map.assert_called_once_with(
        [
            {"name": "Rice", "id": "UUID-1"},
            {"name": "Gmail", "id": "UUID-2"},
        ]
    )
    assert '"persisted": 2' in capsys.readouterr().out


def test_sync_accounts_bootstraps_from_mail(capsys):
    manager = MagicMock()
    manager.save_account_map.return_value = 1
    accounts = [{"name": "Rice", "id": "UUID-1"}]

    with (
        patch(
            "apple_mail_mcp.server.list_accounts",
            new=AsyncMock(return_value=accounts),
        ),
        patch(
            "apple_mail_mcp.index.IndexManager.get_instance",
            return_value=manager,
        ),
    ):
        cli_sync_accounts()

    manager.save_account_map.assert_called_once_with(accounts)
    assert '"persisted": 1' in capsys.readouterr().out


def test_sync_accounts_rejects_malformed_map(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_sync_accounts("Rice")

    assert exc_info.value.code == 1
    assert "display-name=UUID" in capsys.readouterr().err
