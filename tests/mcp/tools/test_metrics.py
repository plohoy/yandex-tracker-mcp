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
from mcp_tracker.tracker.proto.types.refs import UserReference
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
    assignee: str | None = None,
    created_by: str | None = None,
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
        assignee=UserReference.model_construct(display=assignee) if assignee else None,
        created_by=UserReference.model_construct(display=created_by)
        if created_by
        else None,
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

    async def test_metrics_without_queue_filters_by_version_only(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        issues = [_issue("TEST-1", status_type="open")]
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)

        result = await client_session.call_tool(
            "issues_metrics_release_readiness", {"version_id": 1}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["issues_total"] == 1
        assert content["queue"] is None  # no queue field on the mock issue


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
            "issues_metrics_testing_cycle", {"queues": ["TEST"]}
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
        pq = content["per_queue"]["TEST"]
        assert pq["issues_scanned"] == 2
        assert pq["test_cycles_total"] == 3
        assert pq["rework_total"] == 1
        assert pq["in_rework_now"] == 1


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
            {
                "queues": ["TEST"],
                "created_after": "2026-01-01",
                "escape_marker": "прод",
            },
        )

        content = get_tool_result_content(result)
        assert content["created_total"] == 3
        assert content["open_now"] == 2
        assert content["closed_total"] == 1
        assert content["per_queue"]["TEST"]["created_total"] == 3
        assert content["aging_buckets_days"] == {"<30d": 1, "30-90d": 1}
        assert content["escapes"] == 1
        assert content["escape_table"][0]["key"] == "TEST-1"
        assert content["trend"][0]["created"] == 1  # one row per week
        assert sum(row["created"] for row in content["trend"]) == 3

    async def test_trend_org_wide_aggregates(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_queues_protocol: AsyncMock,
    ) -> None:
        from mcp_tracker.tracker.proto.types.queues import Queue

        now = datetime(2026, 8, 21, tzinfo=timezone.utc)
        mock_queues_protocol.queues_list = AsyncMock(
            return_value=[
                Queue.model_construct(key="Q1"),
                Queue.model_construct(key="Q2"),
            ]
        )
        q1 = [_issue("TEST-1", status_type="open", created_at=now)]
        q2: list[Issue] = []
        mock_issues_protocol.issues_find_filter = AsyncMock(side_effect=[q1, q2])

        result = await client_session.call_tool(
            "issues_metrics_defect_trend", {"created_after": "2026-01-01"}
        )

        content = get_tool_result_content(result)
        assert content["queues"] == ["Q1", "Q2"]
        assert content["created_total"] == 1
        assert content["per_queue"] == {
            "Q1": {"created_total": 1, "closed_total": 0, "open_now": 1},
            "Q2": {"created_total": 0, "closed_total": 0, "open_now": 0},
        }


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
            "issues_metrics_data_discipline", {"queues": ["TEST"], "stale_days": 30}
        )

        content = get_tool_result_content(result)
        assert content["totals"]["issues_total"] == 2
        assert content["totals"]["without_estimation"]["count"] == 1
        assert content["totals"]["without_spent"]["count"] == 1
        assert content["totals"]["without_description"]["count"] == 1
        assert content["totals"]["stale_non_final"] == 1
        assert content["per_queue"]["TEST"]["without_estimation"]["count"] == 1
        assert content["stale_table"][0]["key"] == "TEST-1"
        assert content["stale_table"][0]["queue"] == "TEST"


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
            {"queues": ["TEST"], "max_issues": 100},
        )

        content = get_tool_result_content(result)
        assert content["issues_scanned"] == 100
        assert content["coverage"]["complete"] is False


class TestQaDashboard:
    async def test_dashboard_aggregates(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_fields_protocol: AsyncMock,
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
        ]
        ws_issues = [_issue("TEST-5", status="Тестируется")]
        cycle_issues = release
        # call order: release, cycle, defects, discipline, workset
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[
                release,
                cycle_issues,
                defects,
                discipline,
                ws_issues,
            ]
        )
        mock_fields_protocol.get_statuses = AsyncMock(
            return_value=[
                type("S", (), {"key": "qa_active", "name": "Тестируется"})(),
                type("S", (), {"key": "open", "name": "Открыт"})(),
            ]
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
        assert content["defects"]["created_total"] == 1
        assert content["defects"]["escapes"] == 1
        assert content["defects"]["escapes_share"] == 1.0
        assert content["workset"]["total_in_testing"] == 1
        assert content["workset"]["per_queue"]["TEST"]["in_testing"] == 1
        assert content["workset"]["coverage"]["testing_statuses_resolved"] is True
        disc = content["discipline"]["TEST"]
        assert disc["without_estimation"]["count"] == 1
        assert disc["without_spent"]["count"] == 1
        assert disc["without_description"]["count"] == 1
        assert content["cycle"]["issues_with_returns"] == 2
        assert content["cycle"]["return_rate_share"] == 1.0

    async def test_dashboard_without_version_skips_cycle(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_queues_protocol: AsyncMock,
        mock_fields_protocol: AsyncMock,
    ) -> None:
        mock_queues_protocol.queues_get_versions = AsyncMock(return_value=[])
        defects: list[Issue] = []
        discipline: list[Issue] = []
        ws_issues: list[Issue] = []
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[defects, discipline, ws_issues]
        )
        mock_fields_protocol.get_statuses = AsyncMock(
            return_value=[type("S", (), {"key": "qa_active", "name": "Тестируется"})()]
        )

        result = await client_session.call_tool(
            "issues_metrics_qa_dashboard", {"queues": ["TEST"]}
        )

        content = get_tool_result_content(result)
        assert content["release"] is None
        assert content["cycle"] is None
        assert content["defects"]["created_total"] == 0
        assert content["workset"]["total_in_testing"] == 0

    async def test_dashboard_auto_resolves_latest_version(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_queues_protocol: AsyncMock,
        mock_fields_protocol: AsyncMock,
    ) -> None:
        mock_queues_protocol.queues_get_versions = AsyncMock(
            return_value=[
                {"id": 3, "name": "R3", "startDate": "2026-05-01", "released": True},
                {"id": 5, "name": "R5", "startDate": "2026-07-01"},
            ]
        )
        release = [_issue("TEST-1", status_type="open")]
        defects: list[Issue] = []
        discipline: list[Issue] = []
        ws_issues: list[Issue] = []
        # call order: version counts (R5, R3), release, cycle, defects,
        # discipline, workset
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[
                release,
                release,
                release,
                release,
                defects,
                discipline,
                ws_issues,
            ]
        )
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(return_value=[])
        mock_fields_protocol.get_statuses = AsyncMock(return_value=[])

        result = await client_session.call_tool(
            "issues_metrics_qa_dashboard", {"queues": ["TEST"]}
        )

        content = get_tool_result_content(result)
        assert content["release"]["version_id"] == 5  # most loaded among active
        assert content["release"]["version_name"] == "R5"
        assert content["cycle"]["version_id"] == 5

    async def test_dashboard_defaults_to_all_queues_workset(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_queues_protocol: AsyncMock,
        mock_fields_protocol: AsyncMock,
    ) -> None:
        from mcp_tracker.tracker.proto.types.queues import Queue

        mock_queues_protocol.queues_get_versions = AsyncMock(return_value=[])
        mock_queues_protocol.queues_list = AsyncMock(
            return_value=[
                Queue.model_construct(key="Q1"),
                Queue.model_construct(key="Q2"),
            ]
        )
        mock_fields_protocol.get_statuses = AsyncMock(
            return_value=[type("S", (), {"key": "qa_active", "name": "Тестируется"})()]
        )
        defects: list[Issue] = []
        discipline: list[Issue] = []
        ws_q1 = [_issue("TEST-1", status="Тестируется")]
        ws_q2: list[Issue] = []
        # defects(Q1), discipline(Q1), discipline(Q2),
        # workset(Q1), workset(Q2)
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[defects, discipline, discipline, ws_q1, ws_q2]
        )

        result = await client_session.call_tool("issues_metrics_qa_dashboard", {})

        content = get_tool_result_content(result)
        assert content["workset_queues"] == ["Q1", "Q2"]
        assert content["workset"]["total_in_testing"] == 1
        assert content["workset"]["per_queue"]["Q1"]["in_testing"] == 1
        assert content["workset"]["per_queue"]["Q2"]["in_testing"] == 0
        # release/discipline scope falls back to the discovered queues when the
        # release-queues env is unset (no org-specific names in the repo)
        assert set(content["discipline"]) == {"Q1", "Q2"}


class TestQaWorkset:
    async def test_assignee_counts_aggregate_per_tester(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_fields_protocol: AsyncMock,
    ) -> None:
        from mcp_tracker.tracker.proto.types.statuses import Status

        mock_fields_protocol.get_statuses = AsyncMock(
            return_value=[
                Status.model_construct(
                    key="readyForTest", name="Можно тестировать (qa_ready)"
                ),
                Status.model_construct(key="testing", name="Тестируется"),
            ]
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[
                [_issue("TEST-1", assignee="Иван"), _issue("TEST-2", assignee="Пётр")],
                [_issue("TEST-3", assignee="Иван")],
            ]
        )

        result = await client_session.call_tool(
            "issues_list_qa_workset", {"queues": ["TEST"]}
        )

        content = get_tool_result_content(result)
        assert content["assignee_counts"] == {
            "Иван": {"qa_ready": 1, "qa_active": 1, "total": 2},
            "Пётр": {"qa_ready": 1, "qa_active": 0, "total": 1},
        }
        assert content["total_unique"] == 3

    async def test_assignee_filter_unassigned(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_fields_protocol: AsyncMock,
    ) -> None:
        from mcp_tracker.tracker.proto.types.statuses import Status

        mock_fields_protocol.get_statuses = AsyncMock(
            return_value=[
                Status.model_construct(
                    key="readyForTest", name="Можно тестировать (qa_ready)"
                ),
                Status.model_construct(key="testing", name="Тестируется"),
            ]
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[
                [_issue("TEST-1", assignee="Иван"), _issue("TEST-2")],
                [_issue("TEST-3", assignee="Иван")],
            ]
        )

        result = await client_session.call_tool(
            "issues_list_qa_workset",
            {"queues": ["TEST"], "assignee": "не назначен"},
        )

        content = get_tool_result_content(result)
        assert content["total_unique"] == 1
        assert content["returned_total"] == 1
        assert content["table"][0]["key"] == "TEST-2"
        assert content["assignee_counts"] == {
            "не назначен": {"qa_ready": 1, "qa_active": 0, "total": 1}
        }
        assert content["filter"]["assignee"] == "не назначен"


class TestCreatedOpenAggregate:
    async def test_aggregate_groups_by_creator(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        now = datetime(2026, 8, 21, tzinfo=timezone.utc)
        issues = [
            _issue("TEST-1", status_type="open", created_by="Иван", updated_at=now),
            _issue("TEST-2", status_type="open", created_by="Пётр", updated_at=now),
            _issue("TEST-3", status_type="done", created_by="Иван", updated_at=now),
            _issue("TEST-4", status_type="open", created_by="Иван", updated_at=now),
        ]
        mock_issues_protocol.issues_find_filter = AsyncMock(side_effect=[issues, []])

        result = await client_session.call_tool(
            "issues_created_open", {"aggregate": True}
        )

        content = get_tool_result_content(result)
        assert content["scope"] == "all_creators_org_wide_open"
        assert content["drained_issues"] == 3  # TEST-3 closed, excluded
        assert content["top_creators"][0]["creator"] == "Иван"
        assert content["top_creators"][0]["total"] == 2
        assert content["top_creators"][1]["creator"] == "Пётр"
        assert content["top_creators"][1]["total"] == 1


class TestSprintCarryoverAllQueues:
    async def test_all_queues_summary(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_queues_protocol: AsyncMock,
    ) -> None:
        from mcp_tracker.tracker.proto.types.queues import Queue

        mock_queues_protocol.queues_list = AsyncMock(
            return_value=[
                Queue.model_construct(key="Q1"),
                Queue.model_construct(key="Q2"),
            ]
        )
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[
                {"id": 1, "name": "Q1 BOARD"},
                {"id": 2, "name": "OTHER"},
            ]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            return_value=[
                {
                    "id": 11,
                    "name": "S1",
                    "startDate": "2026-07-01",
                    "endDate": "2026-07-14",
                },
                {
                    "id": 12,
                    "name": "S2",
                    "startDate": "2026-07-15",
                    "endDate": "2026-07-28",
                },
            ]
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=[])

        result = await client_session.call_tool(
            "issues_metrics_sprint_carryover", {"all_queues": True}
        )

        content = get_tool_result_content(result)
        assert content["scope"] == "all_queues"
        assert content["totals"]["queues_checked"] == 2
        assert content["per_queue"]["Q1"]["status"] == "complete"
        assert content["per_queue"]["Q2"]["status"] == "board_not_found"
        assert content["per_queue"]["Q1"]["carried_over"] == 0


class TestDisciplineTable:
    def test_fenced_table_has_row_newlines(self) -> None:
        from mcp_tracker.mcp.tools.metrics import _discipline_table

        table = _discipline_table(
            {
                "Q1": {
                    "issues_total": 100,
                    "without_estimation": {"count": 76, "share": 0.762},
                    "without_spent": {"count": 90, "share": 0.9},
                    "without_description": {"count": 20, "share": 0.2},
                    "stale_non_final": 12,
                }
            },
            "Залежались >30д (не финальные)",
            "stale_non_final",
        )
        lines = table.splitlines()
        assert lines[0] == "```"
        assert "76.2% (76)" in table
        assert lines[-1] == "```"
        # every markdown row on its own line (the bug: one collapsed line)
        assert sum(1 for l in lines if l.startswith("|")) == 3
