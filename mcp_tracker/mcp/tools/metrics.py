"""QA-lead metrics tools for Yandex Tracker (grouped by focus area).

Implements the metrics requested by the QA department head. Groups 1
(returns/testing quality) and 5 (team load) are fully covered by existing
tools (issues_count_release_status_returns, issues_summarize_effort,
issues_assigned_open, issues_created_open, issues_count_queue_status,
issues_list_qa_workset); this module adds the missing groups:

- Group 2 (release readiness): issues_metrics_release_readiness
- Group 3 (process efficiency): issues_metrics_testing_cycle
- Group 4 (defects & escapes): issues_metrics_defect_trend
- Group 6 (data discipline):   issues_metrics_data_discipline
- Group 7 (flow / sprints):    issues_metrics_sprint_carryover

All responses follow the fork contract: status/complete/coverage/
reporting_contract, list fields self-capped to the response budget
(rows_capped in coverage), counters always complete.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_tracker.mcp.context import AppContext
from mcp_tracker.mcp.tools.issue_read import (
    _FINAL_STATUS_TYPE_KEYS,
    _cap_rows_for_budget,
    _reference_display,
    _reference_key,
)
from mcp_tracker.mcp.utils import get_yandex_auth
from mcp_tracker.settings import Settings

_TESTING_DISPLAYS = {"тестируется", "можно тестировать"}
_FAIL_DISPLAYS = {"провал"}
_CRITICAL_PRIORITIES = {"critical", "blocker"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hours_between(start: Any, end: Any) -> float:
    """Hours between two ISO timestamps; 0.0 when unparseable."""
    try:
        a = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        if a.tzinfo is None:
            a = a.replace(tzinfo=timezone.utc)
        if b.tzinfo is None:
            b = b.replace(tzinfo=timezone.utc)
        return max(0.0, (b - a).total_seconds() / 3600)
    except (ValueError, TypeError):
        return 0.0


def _bucket_label(value: datetime, group_by: str) -> str:
    if group_by == "month":
        return f"{value.year:04d}-{value.month:02d}"
    iso = value.isocalendar()
    return f"{value.year:04d}-W{iso.week:02d}"


def _age_bucket(age_days: int) -> str:
    if age_days < 30:
        return "<30d"
    if age_days <= 90:
        return "30-90d"
    return ">90d"


async def _drain_filtered(
    issues_api: Any,
    auth: Any,
    filters: dict[str, object],
    fields: list[str],
    max_issues: int,
    log_name: str,
) -> tuple[list[Any], bool]:
    """Paginated issues_find_filter drain (100/page) up to max_issues."""
    issues: list[Any] = []
    seen: set[str] = set()
    page = 1
    while True:
        batch = await issues_api.issues_find_filter(
            filters,
            fields=fields,
            per_page=100,
            page=page,
            auth=auth,
        )
        if not batch:
            return issues, True
        for issue in batch:
            if not issue.key or issue.key in seen:
                raise RuntimeError(f"{log_name}: unstable pagination at page {page}")
            seen.add(issue.key)
            issues.append(issue)
            if len(issues) > max_issues:
                raise ValueError(f"{log_name}: exceeds max_issues={max_issues}")
        if len(batch) < 100:
            return issues, True
        page += 1


def _status_events(changelog: Any) -> list[tuple[str, str, str]]:
    """(updatedAt, from_display, to_display) for status changes, sorted."""
    events: list[tuple[str, str, str]] = []
    for change in changelog or []:
        ts = change.get("updatedAt")
        for changed in change.get("fields", []) or []:
            field = changed.get("field") or {}
            if (field.get("id") or field.get("key")) != "status":
                continue
            events.append(
                (
                    str(ts or ""),
                    _reference_display(changed.get("from")) or "",
                    _reference_display(changed.get("to")) or "",
                )
            )
    events.sort(key=lambda event: event[0])
    return events


def _cycle_metrics(
    events: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Test-cycle and rework metrics from status events.

    test_cycle: entry into a testing status -> exit from it.
    rework: exit to "провал" -> next entry into testing.
    """
    test_cycles = 0
    test_hours = 0.0
    reworks = 0
    rework_hours = 0.0
    in_testing = False
    entry_ts: str | None = None
    fail_ts: str | None = None
    for ts, frm, to in events:
        ft, tt = frm.casefold(), to.casefold()
        if tt in _TESTING_DISPLAYS:
            if fail_ts is not None and not in_testing:
                reworks += 1
                rework_hours += _hours_between(fail_ts, ts)
                fail_ts = None
            if not in_testing:
                in_testing = True
                entry_ts = ts
        elif in_testing and ft in _TESTING_DISPLAYS:
            in_testing = False
            test_cycles += 1
            test_hours += _hours_between(entry_ts, ts)
            if tt in _FAIL_DISPLAYS:
                fail_ts = ts
    return {
        "test_cycles": test_cycles,
        "test_hours_total": round(test_hours, 2),
        "rework_count": reworks,
        "rework_hours_total": round(rework_hours, 2),
        "currently_in_rework": fail_ts is not None,
    }


def register_metrics_tools(settings: Settings, mcp: FastMCP[Any]) -> None:
    @mcp.tool(
        title="Release Readiness Metrics",
        description=(
            "QA-lead release readiness metrics for a release version: issue "
            "composition by status type, open critical/blocker issues, "
            "estimation coverage and defect density. Use for 'готов ли релиз', "
            "'сколько критичных открыто', 'покрытие оценками', "
            "'плотность дефектов'. Counters always cover the whole release; "
            "the open-critical table is capped (rows_capped in coverage)."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_release_readiness(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        version_id: Annotated[int, Field(gt=0, description="Version id")],
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        issues, complete = await _drain_filtered(
            issues_api,
            auth,
            {"queue": queue, "fixVersions": version_id},
            [
                "key",
                "summary",
                "status",
                "statusType",
                "priority",
                "type",
                "estimation",
            ],
            max_issues,
            "release_readiness",
        )
        status_types: Counter[str] = Counter()
        open_critical: list[dict[str, Any]] = []
        with_estimation = 0
        bugs = 0
        for issue in issues:
            st = _reference_key(getattr(issue, "statusType", None))
            status_types[st or "unknown"] += 1
            is_open = st not in _FINAL_STATUS_TYPE_KEYS
            if issue.estimation:
                with_estimation += 1
            type_key = (_reference_key(getattr(issue, "type", None)) or "").casefold()
            if type_key in ("bug", "defect"):
                bugs += 1
            if (
                is_open
                and (_reference_key(getattr(issue, "priority", None)) or "").casefold()
                in _CRITICAL_PRIORITIES
            ):
                open_critical.append(
                    {
                        "key": issue.key,
                        "summary": issue.summary,
                        "priority": _reference_display(
                            getattr(issue, "priority", None)
                        ),
                        "updated_at": (
                            issue.updated_at.isoformat()
                            if hasattr(issue.updated_at, "isoformat")
                            else None
                        ),
                        "url": f"https://tracker.yandex.ru/{issue.key}",
                    }
                )
        open_critical.sort(key=lambda row: str(row["updated_at"] or ""), reverse=True)
        total = len(issues)
        response = {
            "status": "complete",
            "complete": True,
            "queue": queue,
            "version_id": version_id,
            "issues_total": total,
            "status_type_counts": dict(status_types),
            "open_critical": len(open_critical),
            "open_critical_table": open_critical,
            "estimation_coverage": {
                "with_value": with_estimation,
                "without_value": total - with_estimation,
                "share_with_value": round(with_estimation / total, 3)
                if total
                else None,
            },
            "defect_density": {
                "bugs": bugs,
                "share": round(bugs / total, 3) if total else None,
            },
            "coverage": {
                "processed_issues": total,
                "complete": complete,
                "open_definition": "statusType not in done/cancelled",
                "critical_priorities": sorted(_CRITICAL_PRIORITIES),
                "defect_type_keys": ["bug", "defect"],
            },
            "reporting_contract": {
                "state_definition_of_open": True,
                "state_critical_priorities": True,
                "do_not_call_per_issue": True,
            },
        }
        return _cap_rows_for_budget(response, "open_critical_table")

    @mcp.tool(
        title="Testing Cycle Time Metrics",
        description=(
            "QA-lead process metrics from status history: average time in "
            "testing (Тестируется/Можно тестировать) per cycle, rework count "
            "and average rework time (Провал -> next testing entry), and "
            "issues currently in rework (in 'В работе' after a failed test). "
            "Use for 'сколько времени тестируется', 'время реворка', "
            "'что сейчас чинится после провала'. Optionally scope to a "
            "release version. Changelog per issue is cached server-side; "
            "the in-rework table is capped (rows_capped in coverage)."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_testing_cycle(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        version_id: Annotated[
            int | None, Field(gt=0, description="Optional release version id")
        ] = None,
        max_issues: Annotated[int, Field(ge=1, le=2_000)] = 500,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        filters: dict[str, object] = {"queue": queue}
        if version_id is not None:
            filters["fixVersions"] = version_id
        issues, complete = await _drain_filtered(
            issues_api,
            auth,
            filters,
            ["key", "summary", "status", "updatedAt"],
            max_issues,
            "testing_cycle",
        )
        test_cycles = 0
        test_hours = 0.0
        reworks = 0
        rework_hours = 0.0
        with_history = 0
        in_rework_now: list[dict[str, Any]] = []
        for issue in issues:
            changelog = await issues_api.issue_get_status_changelog(
                issue.key, auth=auth
            )
            events = _status_events(changelog)
            if not events:
                continue
            with_history += 1
            metrics = _cycle_metrics(events)
            test_cycles += metrics["test_cycles"]
            test_hours += metrics["test_hours_total"]
            reworks += metrics["rework_count"]
            rework_hours += metrics["rework_hours_total"]
            status_display = (
                (issue.status.display or "") if issue.status else ""
            ).casefold()
            if metrics["currently_in_rework"] and status_display == "в работе":
                in_rework_now.append(
                    {
                        "key": issue.key,
                        "summary": issue.summary,
                        "updated_at": (
                            issue.updated_at.isoformat()
                            if hasattr(issue.updated_at, "isoformat")
                            else None
                        ),
                        "url": f"https://tracker.yandex.ru/{issue.key}",
                    }
                )
        in_rework_now.sort(key=lambda row: str(row["updated_at"] or ""), reverse=True)
        response = {
            "status": "complete",
            "complete": True,
            "queue": queue,
            "version_id": version_id,
            "issues_scanned": len(issues),
            "issues_with_testing_history": with_history,
            "test_cycles_total": test_cycles,
            "avg_test_cycle_hours": round(test_hours / test_cycles, 2)
            if test_cycles
            else None,
            "rework_total": reworks,
            "avg_rework_hours": round(rework_hours / reworks, 2) if reworks else None,
            "in_rework_now": len(in_rework_now),
            "in_rework_table": in_rework_now,
            "testing_statuses": sorted(_TESTING_DISPLAYS),
            "coverage": {
                "processed_issues": len(issues),
                "complete": complete,
                "rework_definition": "Провал -> next entry into testing",
                "cycle_definition": "entry into testing -> exit from it",
                "in_rework_definition": "current status 'В работе' and last testing exit was Провал",
            },
            "reporting_contract": {
                "hours_not_days": True,
                "state_definitions": True,
                "avg_over_empty_is_null": True,
                "do_not_call_issue_get_per_task": True,
            },
        }
        return _cap_rows_for_budget(response, "in_rework_table")

    @mcp.tool(
        title="Defect Trend Metrics",
        description=(
            "QA-lead defect metrics for issues created in a date range: "
            "created/closed/open totals, weekly trend, open-bug aging buckets "
            "(<30 / 30-90 / >90 days), and escapes (issues whose tags or "
            "resolution contain escape_marker, e.g. 'прод' or 'продакшен'). "
            "Use for 'тренд багов', 'старение багов', 'утечки в прод'. "
            "Closed dates are approximated by updated_at (reporting_contract). "
            "The trend table is capped (rows_capped in coverage)."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_defect_trend(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        created_after: Annotated[
            str, Field(description="ISO date, inclusive start of the range")
        ],
        created_before: Annotated[
            str | None, Field(description="ISO date, inclusive end; defaults to today")
        ] = None,
        escape_marker: Annotated[
            str | None,
            Field(
                description="Substring matched case-insensitively against tags and resolution to detect escapes to production"
            ),
        ] = None,
        bug_type_keys: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=10,
                description="Issue type keys counted as bugs",
            ),
        ] = Field(default_factory=lambda: ["bug"]),
        group_by: Annotated[
            Literal["week", "month"], Field(description="Trend bucket")
        ] = "week",
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        end = created_before or _now().date().isoformat()
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        issues, complete = await _drain_filtered(
            issues_api,
            auth,
            {"queue": queue, "type": bug_type_keys, "created": [created_after, end]},
            [
                "key",
                "summary",
                "status",
                "statusType",
                "priority",
                "resolution",
                "tags",
                "createdAt",
                "updatedAt",
            ],
            max_issues,
            "defect_trend",
        )
        closed = 0
        open_now = 0
        created_by_bucket: Counter[str] = Counter()
        closed_by_bucket: Counter[str] = Counter()
        aging: Counter[str] = Counter()
        escapes: list[dict[str, Any]] = []
        today = _now().date()
        for issue in issues:
            st = _reference_key(getattr(issue, "statusType", None))
            is_open = st not in _FINAL_STATUS_TYPE_KEYS
            if is_open:
                open_now += 1
            else:
                closed += 1
            created_at = issue.created_at
            if isinstance(created_at, datetime):
                created_by_bucket[_bucket_label(created_at, group_by)] += 1
            if not is_open and isinstance(issue.updated_at, datetime):
                closed_by_bucket[_bucket_label(issue.updated_at, group_by)] += 1
            if is_open and isinstance(created_at, datetime):
                age_days = (today - created_at.date()).days
                aging[_age_bucket(age_days)] += 1
            if escape_marker:
                marker = escape_marker.casefold()
                resolution = (
                    _reference_display(getattr(issue, "resolution", None))
                    or _reference_key(getattr(issue, "resolution", None))
                    or ""
                )
                haystack = " ".join(issue.tags or []) + " " + resolution
                if marker in haystack.casefold():
                    escapes.append(
                        {
                            "key": issue.key,
                            "summary": issue.summary,
                            "status": issue.status.display if issue.status else None,
                            "priority": _reference_display(
                                getattr(issue, "priority", None)
                            ),
                            "updated_at": (
                                issue.updated_at.isoformat()
                                if hasattr(issue.updated_at, "isoformat")
                                else None
                            ),
                            "url": f"https://tracker.yandex.ru/{issue.key}",
                        }
                    )
        escapes.sort(key=lambda row: str(row["updated_at"] or ""), reverse=True)
        all_buckets = sorted(set(created_by_bucket) | set(closed_by_bucket))
        trend = [
            {
                "period": bucket,
                "created": created_by_bucket.get(bucket, 0),
                "closed": closed_by_bucket.get(bucket, 0),
            }
            for bucket in all_buckets
        ]
        total = len(issues)
        response = {
            "status": "complete",
            "complete": True,
            "queue": queue,
            "range": {"created_after": created_after, "created_before": end},
            "bug_type_keys": bug_type_keys,
            "created_total": total,
            "closed_total": closed,
            "open_now": open_now,
            "aging_buckets_days": dict(aging),
            "escapes": len(escapes),
            "escapes_share": round(len(escapes) / total, 3) if total else None,
            "escape_table": escapes,
            "trend": trend,
            "coverage": {
                "processed_issues": total,
                "complete": complete,
                "closed_date_approximated_by_updated_at": True,
                "escape_marker": escape_marker,
            },
            "reporting_contract": {
                "escape_definition": "tag or resolution contains the marker",
                "closed_dates_are_approximate": True,
                "aging_buckets": ["<30d", "30-90d", ">90d"],
            },
        }
        return _cap_rows_for_budget(response, "escape_table", "trend")

    @mcp.tool(
        title="Tracker Data Discipline Metrics",
        description=(
            "Meta-metrics about tracker data quality in a queue: share of "
            "issues without estimation, without spent time, without "
            "description, and non-final issues stale for more than stale_days "
            "(not updated). Use for 'кто не логирует время', 'без оценки', "
            "'без описания', 'зависшие задачи'. The stale table is capped "
            "(rows_capped in coverage)."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_data_discipline(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        stale_days: Annotated[
            int,
            Field(
                ge=1,
                le=3650,
                description="Non-final issues not updated for this many days are stale",
            ),
        ] = 30,
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        issues, complete = await _drain_filtered(
            issues_api,
            auth,
            {"queue": queue},
            [
                "key",
                "summary",
                "status",
                "statusType",
                "estimation",
                "spent",
                "description",
                "updatedAt",
            ],
            max_issues,
            "data_discipline",
        )
        no_estimation = 0
        no_spent = 0
        no_description = 0
        stale_cutoff = _now() - timedelta(days=stale_days)
        stale: list[dict[str, Any]] = []
        for issue in issues:
            if not issue.estimation:
                no_estimation += 1
            if not issue.spent:
                no_spent += 1
            if not (issue.description or "").strip():
                no_description += 1
            st = _reference_key(getattr(issue, "statusType", None))
            if st not in _FINAL_STATUS_TYPE_KEYS and issue.updated_at is not None:
                try:
                    updated = issue.updated_at
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    if updated < stale_cutoff:
                        stale.append(
                            {
                                "key": issue.key,
                                "summary": issue.summary,
                                "updated_at": updated.isoformat(),
                                "url": f"https://tracker.yandex.ru/{issue.key}",
                            }
                        )
                except (ValueError, TypeError):
                    pass
        stale.sort(key=lambda row: str(row["updated_at"] or ""))
        total = len(issues)
        response = {
            "status": "complete",
            "complete": True,
            "queue": queue,
            "stale_days": stale_days,
            "issues_total": total,
            "without_estimation": {
                "count": no_estimation,
                "share": round(no_estimation / total, 3) if total else None,
            },
            "without_spent": {
                "count": no_spent,
                "share": round(no_spent / total, 3) if total else None,
            },
            "without_description": {
                "count": no_description,
                "share": round(no_description / total, 3) if total else None,
            },
            "stale_non_final": len(stale),
            "stale_table": stale,
            "coverage": {
                "processed_issues": total,
                "complete": complete,
                "stale_definition": f"non-final and not updated for >={stale_days} days",
                "estimation_definition": "issue.estimation present",
                "spent_definition": "issue.spent present (worklog time)",
            },
            "reporting_contract": {
                "shares_are_of_processed_issues": True,
                "spent_is_logged_time_not_lifetime": True,
                "do_not_call_issue_get_per_task": True,
            },
        }
        return _cap_rows_for_budget(response, "stale_table")

    @mcp.tool(
        title="Sprint Carryover Metrics",
        description=(
            "QA-lead flow metric: how much of the previous sprint's issue set "
            "carried over into the current sprint. Resolves the queue's board "
            "(ambiguous board names return candidates), takes the two most "
            "recent sprints by end date, and reports carried-over issues. Use "
            "for 'сколько перенесли между спринтами', 'перегрузка "
            "планирования'. The carried table is capped (rows_capped in "
            "coverage)."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_sprint_carryover(
        ctx: Context[Any, AppContext],
        queue: Annotated[
            str, Field(min_length=1, description="Queue key used to find the board")
        ],
        board_id: Annotated[
            int | None,
            Field(
                gt=0, description="Explicit board id; otherwise resolved by queue name"
            ),
        ] = None,
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        if board_id is None:
            boards = await issues_api.boards_get_all(auth=auth)
            needle = f" {queue.casefold()} "
            candidates = [
                board
                for board in boards
                if needle in f" {str(board.get('name', '')).casefold()} "
            ]
            if len(candidates) != 1:
                return {
                    "status": "ambiguous_board" if candidates else "board_not_found",
                    "complete": False,
                    "queue": queue,
                    "candidate_boards": [
                        {"id": board.get("id"), "name": board.get("name")}
                        for board in candidates
                    ],
                    "required_action": "Supply board_id; do not report a carryover count.",
                }
            board_id = int(candidates[0]["id"])
        sprints = await issues_api.board_get_sprints(board_id, auth=auth)
        dated = [
            sprint
            for sprint in sprints
            if sprint.get("endDate") or sprint.get("end_date")
        ]
        dated.sort(
            key=lambda sprint: str(
                sprint.get("endDate") or sprint.get("end_date") or ""
            ),
            reverse=True,
        )
        if len(dated) < 2:
            return {
                "status": "insufficient_sprints",
                "complete": False,
                "board_id": board_id,
                "sprints_found": len(dated),
                "required_action": "Need at least two dated sprints to compute carryover.",
            }
        current, previous = dated[0], dated[1]
        issues, complete = await _drain_filtered(
            issues_api,
            auth,
            {"queue": queue},
            ["key", "summary", "status", "statusType", "sprint"],
            max_issues,
            "sprint_carryover",
        )
        keys_by_sprint: dict[int, dict[str, Any]] = defaultdict(dict)
        for issue in issues:
            for sprint_ref in issue.sprint or []:
                try:
                    keys_by_sprint[int(getattr(sprint_ref, "id", 0))][issue.key] = issue
                except (TypeError, ValueError):
                    continue
        current_keys = keys_by_sprint.get(int(current.get("id") or 0), {})
        previous_keys = keys_by_sprint.get(int(previous.get("id") or 0), {})
        carried = [
            {
                "key": key,
                "summary": issue.summary,
                "status": issue.status.display if issue.status else None,
                "url": f"https://tracker.yandex.ru/{key}",
            }
            for key, issue in sorted(previous_keys.items())
            if key in current_keys
        ]
        response = {
            "status": "complete",
            "complete": True,
            "queue": queue,
            "board_id": board_id,
            "current_sprint": {
                "id": current.get("id"),
                "name": current.get("name"),
                "end_date": current.get("endDate") or current.get("end_date"),
            },
            "previous_sprint": {
                "id": previous.get("id"),
                "name": previous.get("name"),
                "end_date": previous.get("endDate") or previous.get("end_date"),
            },
            "current_total": len(current_keys),
            "previous_total": len(previous_keys),
            "carried_over": len(carried),
            "carryover_share": round(len(carried) / len(previous_keys), 3)
            if previous_keys
            else None,
            "carried_table": carried,
            "coverage": {
                "processed_issues": len(issues),
                "complete": complete,
                "sprint_definition": "issue.sprint reference contains the sprint id",
            },
            "reporting_contract": {
                "carryover_definition": "issue present in both previous and current sprint",
                "share_denominator_is_previous_total": True,
                "do_not_invent_sprint_dates": True,
            },
        }
        return _cap_rows_for_budget(response, "carried_table")
