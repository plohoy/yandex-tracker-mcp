from unittest.mock import AsyncMock

from mcp.client.session import ClientSession

from mcp_tracker.tracker.proto.types.issues import Issue
from mcp_tracker.tracker.proto.types.queues import QueueVersion
from tests.mcp.conftest import get_tool_result_content


def _version(version_id: int, name: str) -> QueueVersion:
    return QueueVersion.model_construct(
        id=version_id,
        version=1,
        name=name,
        description=None,
        released=False,
        archived=False,
    )


def _issue(key: str) -> Issue:
    return Issue.model_construct(key=key, summary="Test issue")


def _one_qa_rework_cycle() -> list[dict]:
    def event(source: str, target: str, timestamp: str) -> dict:
        return {
            "updatedAt": timestamp,
            "fields": [{
                "field": {"id": "status"},
                "from": {"display": source},
                "to": {"display": target},
            }],
        }

    return [
        event("В работе", "Тестируется", "2026-01-01T10:00:00"),
        event("Тестируется", "Провал", "2026-01-01T12:00:00"),
        event("Провал", "В работе", "2026-01-02T10:00:00"),
        event("В работе", "Тестируется", "2026-01-03T10:00:00"),
    ]


class TestReleaseReturnsAlias:
    async def test_unique_short_release_resolves_and_counts_once(
        self,
        client_session: ClientSession,
        mock_queues_protocol: AsyncMock,
        mock_issues_protocol: AsyncMock,
    ) -> None:
        mock_queues_protocol.queues_get_versions.return_value = [
            _version(161, "7.0.0 iOS SDK"),
            _version(179, "7.1.0 iOS SDK"),
        ]
        mock_issues_protocol.issues_find_filter.return_value = [_issue("YOURQUEUE-1")]
        mock_issues_protocol.issue_get_status_changelog.return_value = (
            _one_qa_rework_cycle()
        )

        response = await client_session.call_tool(
            "issues_count_release_returns_by_name",
            {"queue": "YOURQUEUE", "release": "7.0.0"},
        )

        assert not response.isError
        result = get_tool_result_content(response)
        assert result["status"] == "complete"
        assert result["resolved_release"] == {"id": 161, "name": "7.0.0 iOS SDK"}
        assert result["issues_in_release"] == 1
        assert result["issues_with_returns"] == 1
        assert result["total_returns"] == 1
        assert result["coverage"]["processed_issues"] == 1
        assert result["coverage"]["returned_only"] is True
        assert "evidence" not in result["table"][0]
        mock_queues_protocol.queues_get_versions.assert_awaited_once()
        mock_issues_protocol.issues_find_filter.assert_awaited_once()
        mock_issues_protocol.issue_get_status_changelog.assert_awaited_once()

    async def test_ambiguous_prefix_stops_before_issue_scan(
        self,
        client_session: ClientSession,
        mock_queues_protocol: AsyncMock,
        mock_issues_protocol: AsyncMock,
    ) -> None:
        mock_queues_protocol.queues_get_versions.return_value = [
            _version(161, "7.0.0 iOS SDK"),
            _version(162, "7.0.0 TvisSDK"),
        ]

        response = await client_session.call_tool(
            "issues_count_release_returns_by_name",
            {"queue": "YOURQUEUE", "release": "7.0.0"},
        )

        assert not response.isError
        result = get_tool_result_content(response)
        assert result["status"] == "ambiguous_release"
        assert result["candidates"] == [
            {"id": 161, "name": "7.0.0 iOS SDK"},
            {"id": 162, "name": "7.0.0 TvisSDK"},
        ]
        assert result["coverage"] == {"issues_scanned": 0, "complete": False}
        mock_issues_protocol.issues_find_filter.assert_not_called()
        mock_issues_protocol.issue_get_status_changelog.assert_not_called()

    async def test_exact_full_name_wins_over_longer_prefix(
        self,
        client_session: ClientSession,
        mock_queues_protocol: AsyncMock,
        mock_issues_protocol: AsyncMock,
    ) -> None:
        mock_queues_protocol.queues_get_versions.return_value = [
            _version(161, "7.0.0"),
            _version(162, "7.0.0 iOS SDK"),
        ]
        mock_issues_protocol.issues_find_filter.return_value = []

        response = await client_session.call_tool(
            "issues_count_release_returns_by_name",
            {"queue": "YOURQUEUE", "release": "7.0.0"},
        )

        assert not response.isError
        result = get_tool_result_content(response)
        assert result["resolved_release"] == {"id": 161, "name": "7.0.0"}
        assert result["issues_in_release"] == 0
        mock_issues_protocol.issues_find_filter.assert_awaited_once()
