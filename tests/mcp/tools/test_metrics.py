"""Tests for the QA-lead metrics tools (mcp_tracker.mcp.tools.metrics)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from mcp.client.session import ClientSession

from mcp_tracker.tracker.proto.types.issues import (
    Issue,
    IssueTypeReference,
    PriorityReference,
    SprintReference,
    StatusReference,
)
from tests.mcp.conftest import get_tool_result_content


def _issue(
    key: str,
    *,
    status: str = "open",
    status_type: str = "open",
    priority: str = "normal",
    type_key: str = "task",
    estimation: str | None = None,
    spent: str | None = None,
    description: str | None = "desc",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    tags: list[str] | None = None,
    sprints: list[int] | None = None,
    resolution: str | None = None,
) -> Issue:
    return Issue.model_construct(
        id=key,
        version=1,
        key=key,
        summary=f"Summary {key}",
        description=description,
        status=StatusReference.model_construct(
            id="1", key=status, display=status.title()
        ),
        statusType=StatusReference.model_construct(
            id="1", key=status_type, display=status_type.title()
        ),
        priority=PriorityReference.model_construct(
            id="3", key=priority, display=priority.title()
        ),
        type=IssueTypeReference.model_construct(
            id="1", key=type_key, display=type_key.title()
        ),
        estimation=estimation,
        spent=spent,
        tags=tags,
        created_at=created_at,
        updated_at=updated_at,
        sprint=[
            SprintReference.model_construct(id=sid, display=f"S{sid}")
            for sid in (sprints or [])
        ],
        resolution=resolution,
    )


def _status_event(ts: str, frm: str, to: str) -> dict:
    return {
        "updatedAt": ts,
        "fields": [
            {
                "field": {"id": "status", "key": "status"},
                "from": {"display": frm},
                "to": {"display": to},
            }
        ],
    }


class TestReleaseReadiness:
    async def test_metrics_computed(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        issues = [
            _issue(
                "TEST-1",
                status_type="open",
                priority="blocker",
                type_key="bug",
                estimation="P1D",
            ),
            _issue("TEST-2", status_type="open", priority="normal"),
            _issue("TEST-3", status_type="done", priority="normal"),
        ]
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)

        result = await client_session.call_tool(
            "issues_metrics_release_readiness",
            {"queue": "TEST", "version_id": 1},
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["issues_total"] == 3
        assert content["status_type_counts"] == {"open": 2, "done": 1}
        assert content["open_critical"] == 1
        assert content["open_critical_table"][0]["key"] == "TEST-1"
        assert content["estimation_coverage"]["with_value"] == 1
        assert content["estimation_coverage"]["share_with_value"] == round(1 / 3, 3)
        assert content["defect_density"]["bugs"] == 1


class TestTestingCycle:
    async def test_cycles_and_rework(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        # TEST-1: two full cycles + one rework, ends closed
        t1_events = [
            _status_event("2026-01-01T10:00:00+00:00", "В работе", "Тестируется"),
            _status_event("2026-01-01T14:00:00+00:00", "Тестируется", "Провал"),
            _status_event("2026-01-02T10:00:00+00:00", "Провал", "В работе"),
            _status_event("2026-01-02T14:00:00+00:00", "В работе", "Тестируется"),
            _status_event("2026-01-02T18:00:00+00:00", "Тестируется", "Готово"),
        ]
        # TEST-2: in rework now (status В работе, last exit was Провал)
        t2_events = [
            _status_event("2026-01-01T10:00:00+00:00", "В работе", "Тестируется"),
            _status_event("2026-01-01T12:00:00+00:00", "Тестируется", "Провал"),
            _status_event("2026-01-01T14:00:00+00:00", "Провал", "В работе"),
        ]
        issues = [
            _issue("TEST-1", status="Готово", status_type="done"),
            _issue("TEST-2", status="В работе", status_type="open"),
        ]
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            side_effect=[t1_events, t2_events]
        )

        result = await client_session.call_tool(
            "issues_metrics_testing_cycle", {"queue": "TEST"}
        )

        content = get_tool_result_content(result)
        assert content["issues_scanned"] == 2
        assert content["issues_with_testing_history"] == 2
        assert content["test_cycles_total"] == 3  # 2 (TEST-1) + 1 (TEST-2)
        assert content["avg_test_cycle_hours"] == round((4 + 4 + 2) / 3, 2)
        assert content["rework_total"] == 1
        assert content["avg_rework_hours"] == 24.0  # 01-01T14:00 -> 01-02T14:00
        assert content["in_rework_now"] == 1
        assert content["in_rework_table"][0]["key"] == "TEST-2"


class TestDefectTrend:
    async def test_trend_aging_escapes(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        now = datetime(2026, 8, 21, tzinfo=timezone.utc)
        issues = [
            _issue(
                "TEST-1",
                status_type="open",
                created_at=now,
                tags=["прод"],
                resolution="Найдено в проде",
            ),
            _issue("TEST-2", status_type="open", created_at=now - timedelta(days=60)),
            _issue("TEST-3", status_type="done", created_at=now - timedelta(days=100)),
        ]
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)

        result = await client_session.call_tool(
            "issues_metrics_defect_trend",
            {"queue": "TEST", "created_after": "2026-01-01", "escape_marker": "прод"},
        )

        content = get_tool_result_content(result)
        assert content["created_total"] == 3
        assert content["open_now"] == 2
        assert content["closed_total"] == 1
        assert content["aging_buckets_days"] == {"<30d": 1, "30-90d": 1}
        assert content["escapes"] == 1
        assert content["escape_table"][0]["key"] == "TEST-1"
        assert content["trend"][0]["created"] == 1  # one row per week
        assert sum(row["created"] for row in content["trend"]) == 3


class TestDataDiscipline:
    async def test_shares_and_stale(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        now = datetime(2026, 8, 21, tzinfo=timezone.utc)
        issues = [
            _issue(
                "TEST-1",
                estimation=None,
                spent=None,
                description="",
                updated_at=now - timedelta(days=60),
            ),
            _issue("TEST-2", estimation="P1D", spent="P1D", updated_at=now),
        ]
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)

        result = await client_session.call_tool(
            "issues_metrics_data_discipline", {"queue": "TEST", "stale_days": 30}
        )

        content = get_tool_result_content(result)
        assert content["issues_total"] == 2
        assert content["without_estimation"]["count"] == 1
        assert content["without_spent"]["count"] == 1
        assert content["without_description"]["count"] == 1
        assert content["stale_non_final"] == 1
        assert content["stale_table"][0]["key"] == "TEST-1"


class TestSprintCarryover:
    async def test_carryover_between_two_sprints(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
    ) -> None:
        boards = [{"id": 7, "name": "TEST Board"}]
        sprints = [
            {"id": 2, "name": "S2", "endDate": "2026-08-15"},
            {"id": 1, "name": "S1", "endDate": "2026-07-15"},
        ]
        issues = [
            _issue("TEST-1", sprints=[2]),
            _issue("TEST-2", sprints=[1]),
            _issue("TEST-3", sprints=[1, 2]),  # carried
            _issue("TEST-4", sprints=[1, 2]),  # carried
        ]
        mock_issues_protocol.boards_get_all = AsyncMock(return_value=boards)
        mock_issues_protocol.board_get_sprints = AsyncMock(return_value=sprints)
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)

        result = await client_session.call_tool(
            "issues_metrics_sprint_carryover", {"queue": "TEST"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["current_total"] == 3  # TEST-1, TEST-3, TEST-4
        assert content["previous_total"] == 3  # TEST-2, TEST-3, TEST-4
        assert content["carried_over"] == 2
        assert content["carryover_share"] == round(2 / 3, 3)
        assert {row["key"] for row in content["carried_table"]} == {"TEST-3", "TEST-4"}

    async def test_ambiguous_board_returns_candidates(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        boards = [{"id": 1, "name": "TEST Old"}, {"id": 2, "name": "TEST New"}]
        mock_issues_protocol.boards_get_all = AsyncMock(return_value=boards)

        result = await client_session.call_tool(
            "issues_metrics_sprint_carryover", {"queue": "TEST"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "ambiguous_board"
        assert content["complete"] is False
        assert len(content["candidate_boards"]) == 2

    async def test_board_not_found_lists_all_boards(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        boards = [{"id": 7, "name": "Доска проекта AndroidSDK"}]
        mock_issues_protocol.boards_get_all = AsyncMock(return_value=boards)

        result = await client_session.call_tool(
            "issues_metrics_sprint_carryover", {"queue": "TEST"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "board_not_found"
        assert content["all_boards"] == [{"id": 7, "name": "Доска проекта AndroidSDK"}]


class TestSampling:
    async def test_drain_samples_over_max_issues(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        # 250 issues in two 100-row pages + one 50-row page; max_issues=100
        # stops the drain at 100 and reports complete=False.
        page1 = [_issue(f"TEST-{i}") for i in range(1, 101)]
        page2 = [_issue(f"TEST-{i}") for i in range(101, 201)]
        page3 = [_issue(f"TEST-{i}") for i in range(201, 251)]
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[page1, page2, page3]
        )
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(return_value=[])

        result = await client_session.call_tool(
            "issues_metrics_testing_cycle",
            {"queue": "TEST", "max_issues": 100},
        )

        content = get_tool_result_content(result)
        assert content["issues_scanned"] == 100
        assert content["coverage"]["complete"] is False


class TestQaDashboard:
    async def test_dashboard_aggregates(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        now = datetime(2026, 8, 21, tzinfo=timezone.utc)
        release = [
            _issue(
                "TEST-1",
                status_type="open",
                priority="blocker",
                type_key="bug",
                estimation="P1D",
            ),
            _issue("TEST-2", status_type="done"),
        ]
        defects = [_issue("TEST-3", status_type="open", created_at=now, tags=["прод"])]
        discipline = [
            _issue("TEST-1", estimation=None, spent=None, description=""),
            _issue("TEST-2", estimation="P1D", spent="P1D", status_type="done"),
            _issue("TEST-4", status="Тестируется", status_type="open"),
        ]
        cycle_issues = release
        # call order: release drain, cycle drain, per-queue defects, discipline
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[release, cycle_issues, defects, discipline]
        )
        changelog = [
            _status_event("2026-01-01T10:00:00+00:00", "В работе", "Тестируется"),
            _status_event("2026-01-01T12:00:00+00:00", "Тестируется", "Провал"),
            _status_event("2026-01-02T10:00:00+00:00", "Провал", "Тестируется"),
        ]
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            return_value=changelog
        )

        result = await client_session.call_tool(
            "issues_metrics_qa_dashboard",
            {
                "queues": ["TEST"],
                "version_id": 1,
                "created_after": "2026-08-01",
                "escape_marker": "прод",
            },
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["release"]["issues_total"] == 2
        assert content["release"]["open_critical"] == 1
        block = content["per_queue"]["TEST"]
        assert block["defects"]["created_total"] == 1
        assert block["defects"]["escapes"] == 1
        assert block["defects"]["escapes_share"] == 1.0
        assert block["workset"]["in_testing"] == 1
        assert block["discipline"]["without_estimation"]["count"] == 2
        assert block["discipline"]["without_spent"]["count"] == 2
        assert block["discipline"]["without_description"]["count"] == 1
        assert content["cycle"]["issues_with_returns"] == 2
        assert content["cycle"]["rework_total"] == 2
        assert content["cycle"]["return_rate_share"] == 1.0

    async def test_dashboard_without_version_skips_cycle(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_queues_protocol: AsyncMock,
    ) -> None:
        mock_queues_protocol.queues_get_versions = AsyncMock(return_value=[])
        defects: list[Issue] = []
        discipline: list[Issue] = []
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[defects, discipline]
        )

        result = await client_session.call_tool(
            "issues_metrics_qa_dashboard", {"queues": ["TEST"]}
        )

        content = get_tool_result_content(result)
        assert content["release"] is None
        assert content["cycle"] is None
        assert content["per_queue"]["TEST"]["defects"]["created_total"] == 0

    async def test_dashboard_auto_resolves_latest_version(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_queues_protocol: AsyncMock,
    ) -> None:
        mock_queues_protocol.queues_get_versions = AsyncMock(
            return_value=[
                {"id": 3, "name": "R3", "startDate": "2026-05-01"},
                {"id": 5, "name": "R5", "startDate": "2026-07-01"},
            ]
        )
        release = [_issue("TEST-1", status_type="open")]
        defects: list[Issue] = []
        discipline: list[Issue] = []
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[release, release, defects, discipline]
        )
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(return_value=[])

        result = await client_session.call_tool(
            "issues_metrics_qa_dashboard", {"queues": ["TEST"]}
        )

        content = get_tool_result_content(result)
        assert content["release"]["version_id"] == 5  # latest by startDate
        assert content["release"]["version_name"] == "R5"
        assert content["cycle"]["version_id"] == 5
