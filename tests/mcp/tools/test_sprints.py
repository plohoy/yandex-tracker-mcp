"""Tests for the board-sprint tools (mcp_tracker.mcp.tools.sprints)."""

from datetime import date, timedelta
from unittest.mock import AsyncMock

from mcp.client.session import ClientSession

from mcp_tracker.tracker.custom.errors import SprintNotFound
from mcp_tracker.tracker.proto.types.issues import Issue, StatusReference
from mcp_tracker.tracker.proto.types.refs import UserReference
from tests.mcp.conftest import get_tool_result_content


def _issue(
    key: str,
    *,
    status: str = "open",
    status_display: str | None = None,
    status_type: str = "open",
    estimation: str | None = None,
    spent: str | None = None,
    assignee: str | None = None,
    queue: str = "TEST",
) -> Issue:
    return Issue.model_construct(
        id=key,
        version=1,
        key=key,
        summary=f"Summary {key}",
        status=StatusReference.model_construct(
            id="1", key=status, display=status_display or status.title()
        ),
        statusType=StatusReference.model_construct(
            id="1", key=status_type, display=status_type.title()
        ),
        estimation=estimation,
        spent=spent,
        assignee=UserReference.model_construct(display=assignee) if assignee else None,
        queue={"id": "1", "key": queue, "display": queue},
    )


def _changelog(*pairs: tuple[str, str | None]) -> list[dict[str, object]]:
    """Status changelog entries from (from, to) display pairs."""
    out: list[dict[str, object]] = []
    for index, (source, target) in enumerate(pairs):
        out.append(
            {
                "updatedAt": f"2026-09-{10 + index:02d}T10:00:00Z",
                "fields": [
                    {
                        "field": {"id": "status"},
                        "from": {"display": source} if source else None,
                        "to": {"display": target} if target else None,
                    }
                ],
            }
        )
    return out


REWORK_CYCLE = _changelog(
    ("Тестируется", "Провал"),
    ("Провал", "В работе"),
    ("В работе", "Тестируется"),
)


def _sprint(
    sprint_id: int,
    *,
    name: str = "S",
    start: str | None = None,
    end: str | None = None,
    state: str = "archived",
) -> dict[str, object]:
    return {
        "id": sprint_id,
        "name": name,
        "startDate": start,
        "endDate": end,
        "status": state,
        "board": {"id": "7", "display": "TEST Board"},
    }


class TestListSprints:
    async def test_resolve_single_sprint_by_id(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(
            return_value=_sprint(
                552, name="Sprint 25", start="2026-09-03", end="2026-09-16"
            )
        )

        result = await client_session.call_tool(
            "issues_list_sprints", {"sprint_id": 552}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["mode"] == "sprint_id"
        assert content["sprint"]["id"] == 552
        assert content["sprint"]["board_id"] == "7"
        assert content["sprint"]["start_date"] == "2026-09-03"
        assert content["is_current"] is False

    async def test_missing_sprint_id_is_honest(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(side_effect=SprintNotFound(999))

        result = await client_session.call_tool(
            "issues_list_sprints", {"sprint_id": 999}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "sprint_not_found"
        assert "required_action" in content

    async def test_board_listing_newest_first_with_limit(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[{"id": 7, "name": "TEST Board"}]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            return_value=[
                _sprint(1, name="old", start="2026-01-01", end="2026-01-14"),
                _sprint(3, name="newest", start="2026-02-01", end="2026-02-14"),
                _sprint(2, name="middle", start="2026-01-15", end="2026-01-28"),
            ]
        )

        result = await client_session.call_tool(
            "issues_list_sprints", {"queue": "TEST", "limit": 2}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert [row["id"] for row in content["sprints"]] == [3, 2]
        assert content["counts"]["sprints_total"] == 3
        assert content["coverage"]["rows_capped"] is True
        assert content["board_id"] == 7

    async def test_ambiguous_board_returns_candidates(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[
                {"id": 1, "name": "TEST Old"},
                {"id": 2, "name": "TEST New"},
            ]
        )

        result = await client_session.call_tool(
            "issues_list_sprints", {"queue": "TEST"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "ambiguous_board"
        assert len(content["candidate_boards"]) == 2

    async def test_board_name_longer_than_board_resolves_in_one_call(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        """A sprint title names the board: board_name resolves and lists sprints."""
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[
                {"id": 179, "name": "Product QA Sprint"},
                {"id": 47, "name": "Unrelated"},
            ]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            return_value=[
                _sprint(
                    552,
                    name="Product QA Sprint 25",
                    start="2026-09-03",
                    end="2026-09-16",
                )
            ]
        )

        result = await client_session.call_tool(
            "issues_list_sprints",
            {"board_name": "Product QA Sprint 25 (03.09 - 16.09)", "limit": 30},
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["board_id"] == 179
        assert [row["id"] for row in content["sprints"]] == [552]

    async def test_board_name_without_match_returns_catalog(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[{"id": 1, "name": "Something Else"}]
        )

        result = await client_session.call_tool(
            "issues_list_sprints", {"board_name": "No Such Board"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "board_not_found"
        assert content["board_name"] == "No Such Board"
        assert content["all_boards"] == [{"id": 1, "name": "Something Else"}]

    async def test_board_without_sprints_offers_stem_candidates(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[
                {"id": 40, "name": "Testteam QA"},
                {"id": 221, "name": "Test Sprint"},
                {"id": 999, "name": "Unrelated"},
            ]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            side_effect=RuntimeError("400 Bad Request")
        )

        result = await client_session.call_tool(
            "issues_list_sprints", {"queue": "TESTTEAM"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "board_without_sprints"
        assert {row["id"] for row in content["candidate_boards"]} == {40, 221}

    async def test_include_future_false_drops_upcoming_sprints(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        today = date.today()
        future_start = (today + timedelta(days=10)).isoformat()
        past_start = (today - timedelta(days=20)).isoformat()
        past_end = (today - timedelta(days=7)).isoformat()
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[{"id": 7, "name": "TEST Board"}]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            return_value=[
                _sprint(
                    11, start=future_start, end=(today + timedelta(days=24)).isoformat()
                ),
                _sprint(10, start=past_start, end=past_end),
            ]
        )

        result = await client_session.call_tool(
            "issues_list_sprints", {"queue": "TEST", "include_future": False}
        )

        content = get_tool_result_content(result)
        assert [row["id"] for row in content["sprints"]] == [10]


class TestSprintResults:
    async def test_counts_hours_breakdown_and_table(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(
            return_value=_sprint(
                552, name="Sprint 25", start="2026-09-03", end="2026-09-16"
            )
        )
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            return_value=REWORK_CYCLE
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[
                _issue(
                    "TEST-1",
                    status="closed",
                    status_display="Закрыт",
                    status_type="done",
                    estimation="P1W",
                    spent="P3D",
                    assignee="Ivan Ivanov",
                ),
                _issue(
                    "TEST-2",
                    status="testing",
                    status_display="Тестируется",
                    status_type="inProgress",
                    estimation="PT1H30M",
                    spent="PT1H30M",
                    assignee="Ivan Ivanov",
                ),
                _issue(
                    "TEST-3",
                    status="cancelled",
                    status_display="Отменён",
                    status_type="cancelled",
                    assignee=None,
                ),
            ]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 552}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["counts"]["issues_total"] == 3
        assert content["counts"]["done"] == 2  # done + cancelled
        assert content["counts"]["remaining"] == 1
        assert content["hours"]["plan_hours"] == 41.5  # 40 + 1.5
        assert content["hours"]["fact_hours"] == 25.5  # 24 + 1.5
        assert content["counts"]["queue_counts"] == {"TEST": 3}
        assert content["counts"]["issues_without_assignee"] == 1
        # per-issue table, exact user-facing shape
        table = content["table"]
        header = table.splitlines()[0]
        assert header == (
            "| Очередь | Номер Задачи | Заголовок | Статус | Исполнитель | План часов "
            "| Факт часов | Количество возвратов |"
        )
        assert table.splitlines()[1] == "|---|---|---|---|---|---|---|---|"
        assert (
            "| TEST | TEST-1 | Summary TEST-1 | Закрыт | Ivan Ivanov | 40 | 24 | 1 |"
            in table
        )
        assert "| TEST | TEST-3 | Summary TEST-3 | Отменён | — | — | — | 1 |" in table
        assert content["coverage"]["table_rows_returned"] == 3
        # returns come from each issue's status changelog
        assert content["counts"]["returns_total"] == 3
        assert content["counts"]["issues_with_returns"] == 3
        assert content["counts"]["returns_metric"] == "qa_rework_cycle"
        assert content["coverage"]["returns_scanned"] == 3
        assert content["coverage"]["returns_skipped"] == 0
        assert content["coverage"]["returns_failed"] == 0

    async def test_long_summaries_trim_the_table_not_the_counters(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(return_value=[])
        long_issues = []
        for index in range(120):
            issue = _issue(f"TEST-{index}")
            issue.summary = "Очень длинный заголовок задачи " * 4
            long_issues.append(issue)
        # page 1 returns the batch, page 2 ends the drain
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[long_issues, []]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 1, "max_return_scans": 0}
        )

        content = get_tool_result_content(result)
        coverage = content["coverage"]
        assert content["counts"]["issues_total"] == 120  # counters stay complete
        assert coverage["table_trimmed_by_size"] is True
        assert coverage["table_rows_returned"] < coverage["table_rows_total"]
        assert coverage["table_rows_total"] == 120
        assert "never shorten" in content["table_note"]

    async def test_returns_column_marks_unscanned_rows(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            return_value=REWORK_CYCLE
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1"), _issue("TEST-2"), _issue("TEST-3")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 1, "max_return_scans": 1}
        )

        content = get_tool_result_content(result)
        assert content["coverage"]["returns_scanned"] == 1
        assert content["coverage"]["returns_skipped"] == 2
        assert content["counts"]["returns_total"] == 1
        rows = {row["key"]: row["returns"] for row in content["rows"]}
        assert rows == {"TEST-1": 1, "TEST-2": None, "TEST-3": None}
        table = content["table"]
        assert "| TEST | TEST-2 | Summary TEST-2 | Open | — | — | — | — |" in table
        assert (
            "unscanned" in content["coverage"]["returns_note"]
            or "NOT scanned" in content["coverage"]["returns_note"]
        )

    async def test_failed_changelog_is_isolated(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        async def changelog(issue_id: str, *, auth: object = None) -> list[dict]:
            if issue_id == "TEST-2":
                raise RuntimeError("504")
            return REWORK_CYCLE

        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            side_effect=changelog
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1"), _issue("TEST-2")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 1}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["coverage"]["returns_failed"] == 1
        assert content["coverage"]["returns_scanned"] == 1
        assert content["counts"]["returns_total"] == 1
        rows = {row["key"]: row["returns"] for row in content["rows"]}
        assert rows == {"TEST-1": 1, "TEST-2": None}

    async def test_returns_metric_can_be_switched(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        # a single Тестируется→Провал transition: 1 for testing_rework, 0 cycles
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            return_value=_changelog(("Тестируется", "Провал"))
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1", status_type="done")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results",
            {"sprint_id": 1, "returns_metric": "testing_rework"},
        )

        content = get_tool_result_content(result)
        assert content["counts"]["returns_metric"] == "testing_rework"
        assert content["counts"]["returns_total"] == 1
        assert content["rows"][0]["returns"] == 1

    async def test_returns_skipped_entirely_when_cap_is_zero(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 1, "max_return_scans": 0}
        )

        content = get_tool_result_content(result)
        assert content["counts"]["returns_metric"] is None
        assert content["coverage"]["returns_scanned"] == 0
        assert (
            "| TEST | TEST-1 | Summary TEST-1 | Open | — | — | — | — |"
            in content["table"]
        )

    async def test_uncapped_breakdown_is_sorted_by_load(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[
                _issue("TEST-1", assignee="One"),
                _issue("TEST-2", assignee="Two"),
                _issue("TEST-3", assignee="Two"),
            ]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 1, "include_table": False}
        )

        content = get_tool_result_content(result)
        assert "table" not in content
        assert [row["assignee"] for row in content["assignee_breakdown"]] == [
            "Two",
            "One",
        ]
        assert content["assignee_breakdown"][0]["issues_total"] == 2

    async def test_sample_reports_incomplete_coverage(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1"), _issue("TEST-2"), _issue("TEST-3")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 1, "max_issues": 2}
        )

        content = get_tool_result_content(result)
        assert content["complete"] is False
        assert content["coverage"]["complete"] is False
        assert "reason" in content["coverage"]

    async def test_sprint_not_found(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(side_effect=SprintNotFound(7))

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 7}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "sprint_not_found"

    async def test_unparsed_duration_is_counted(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.sprint_get = AsyncMock(return_value=_sprint(1))
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1", estimation="two weeks")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_results", {"sprint_id": 1}
        )

        content = get_tool_result_content(result)
        assert content["coverage"]["unparsed_duration_values"] == 1


class TestSprintHistory:
    async def test_board_last_n_totals_and_trend(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        today = date.today()
        sprints = [
            _sprint(
                3,
                name="current",
                start=(today - timedelta(days=3)).isoformat(),
                end=(today + timedelta(days=10)).isoformat(),
                state="in_progress",
            ),
            _sprint(
                2,
                name="previous",
                start=(today - timedelta(days=20)).isoformat(),
                end=(today - timedelta(days=7)).isoformat(),
            ),
            _sprint(
                1,
                name="older",
                start=(today - timedelta(days=40)).isoformat(),
                end=(today - timedelta(days=27)).isoformat(),
            ),
            _sprint(
                4,
                name="planned",
                start=(today + timedelta(days=11)).isoformat(),
                end=(today + timedelta(days=24)).isoformat(),
                state="draft",
            ),
        ]
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[{"id": 7, "name": "TEST Board"}]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(return_value=sprints)
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[
                [_issue("TEST-1", status_type="done", estimation="PT8H", spent="PT4H")],
                [_issue("TEST-2", estimation="PT4H")],
                [],
            ]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_history", {"queue": "TEST", "last_n": 3}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["mode"] == "board_last_n"
        # newest first selection, in-progress sprint included, planned sprint excluded
        assert [row["id"] for row in content["per_sprint"]] == [1, 2, 3]
        assert [row["sprint_id"] for row in content["trend"]] == [1, 2, 3]
        assert content["totals"]["sprints_analysed"] == 3
        assert content["totals"]["issues_total"] == 2
        assert content["totals"]["done_total"] == 1
        assert content["totals"]["plan_hours"] == 12.0
        assert content["totals"]["fact_hours"] == 4.0
        table = content["history_table"]
        assert table.splitlines()[0] == (
            "| Спринт | Период | Задач | Выполнено | Осталось | План | Факт | Выполнено % |"
        )
        assert "current (#3)" in table

    async def test_explicit_ids_isolate_a_failing_sprint(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        async def sprint_get(
            sprint_id: int, *, auth: object = None
        ) -> dict[str, object]:
            if sprint_id == 99:
                raise SprintNotFound(99)
            return _sprint(sprint_id, name=f"S{sprint_id}")

        mock_issues_protocol.sprint_get = AsyncMock(side_effect=sprint_get)
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1", status_type="done")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_history", {"sprint_ids": [10, 99]}
        )

        content = get_tool_result_content(result)
        statuses = {row["id"]: row.get("status") for row in content["per_sprint"]}
        assert statuses[10] == "complete"
        assert statuses[99] == "sprint_not_found"
        assert content["totals"]["sprints_analysed"] == 1
        assert content["totals"]["sprints_failed"] == 1

    async def test_insufficient_sprints_when_nothing_is_in_scope(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        today = date.today()
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[{"id": 7, "name": "TEST Board"}]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            return_value=[
                _sprint(
                    42,
                    name="planned",
                    start=(today + timedelta(days=5)).isoformat(),
                    end=(today + timedelta(days=18)).isoformat(),
                    state="draft",
                )
            ]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_history", {"queue": "TEST", "include_current": False}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "insufficient_sprints"
        assert content["sprints_found"] == 1

    async def test_exclude_current_keeps_finished_sprints(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        today = date.today()
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[{"id": 7, "name": "TEST Board"}]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            return_value=[
                _sprint(
                    3,
                    start=(today - timedelta(days=3)).isoformat(),
                    end=(today + timedelta(days=10)).isoformat(),
                    state="in_progress",
                ),
                _sprint(
                    2,
                    start=(today - timedelta(days=20)).isoformat(),
                    end=(today - timedelta(days=7)).isoformat(),
                ),
            ]
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=[])

        result = await client_session.call_tool(
            "issues_metrics_sprint_history", {"queue": "TEST", "include_current": False}
        )

        content = get_tool_result_content(result)
        assert [row["id"] for row in content["per_sprint"]] == [2]

    async def test_board_without_sprints_offers_candidates(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[
                {"id": 40, "name": "Testteam QA"},
                {"id": 221, "name": "Test Sprint"},
            ]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(return_value=[])

        result = await client_session.call_tool(
            "issues_metrics_sprint_history", {"queue": "TESTTEAM"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "board_without_sprints"
        assert len(content["candidate_boards"]) == 2

    async def test_board_name_resolves_history_scope(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        today = date.today()
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[{"id": 179, "name": "Product QA Sprint"}]
        )
        mock_issues_protocol.board_get_sprints = AsyncMock(
            return_value=[
                _sprint(
                    552,
                    name="Product QA Sprint 25",
                    start=(today - timedelta(days=14)).isoformat(),
                    end=(today - timedelta(days=1)).isoformat(),
                )
            ]
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            return_value=[_issue("TEST-1", status_type="done")]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_history",
            {"board_name": "Product QA Sprint 25", "last_n": 1},
        )

        content = get_tool_result_content(result)
        assert content["status"] == "complete"
        assert content["board_id"] == 179
        assert [row["id"] for row in content["per_sprint"]] == [552]

    async def test_ambiguous_board_returns_candidates(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        mock_issues_protocol.boards_get_all = AsyncMock(
            return_value=[
                {"id": 1, "name": "TEST Old"},
                {"id": 2, "name": "TEST New"},
            ]
        )

        result = await client_session.call_tool(
            "issues_metrics_sprint_history", {"queue": "TEST"}
        )

        content = get_tool_result_content(result)
        assert content["status"] == "ambiguous_board"
        assert len(content["candidate_boards"]) == 2
