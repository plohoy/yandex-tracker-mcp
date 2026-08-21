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

import asyncio
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
    """Paginated issues_find_filter drain (100/page) up to max_issues.

    Returns (issues, complete). When the queue holds more than max_issues
    the drain stops at the limit and complete=False (a sample — callers
    report it in coverage); unstable pagination still fails closed.
    """
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
            if len(issues) >= max_issues:
                return issues, False
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
        version_id: Annotated[int, Field(gt=0, description="Version id")],
        queue: Annotated[
            str | None,
            Field(
                description="Queue key; optional — a version belongs to one queue, resolved from its issues"
            ),
        ] = None,
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if queue is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        filters: dict[str, object] = {"fixVersions": version_id}
        if queue is not None:
            filters["queue"] = queue
        issues, complete = await _drain_filtered(
            issues_api,
            auth,
            filters,
            [
                "key",
                "summary",
                "queue",
                "status",
                "statusType",
                "priority",
                "type",
                "estimation",
            ],
            max_issues,
            "release_readiness",
        )
        resolved_queue = queue or (
            _reference_display(getattr(issues[0], "queue", None))
            or _reference_key(getattr(issues[0], "queue", None))
            if issues
            else None
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
            "queue": resolved_queue,
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
        queues: Annotated[
            list[str] | None,
            Field(
                max_length=10,
                description="Queue keys; default: ALL queues (org-wide sample up to max_issues in total). When queues are given, max_issues applies PER QUEUE.",
            ),
        ] = None,
        version_id: Annotated[
            int | None, Field(gt=0, description="Optional release version id")
        ] = None,
        max_issues: Annotated[int, Field(ge=1, le=2_000)] = 500,
    ) -> dict[str, Any]:
        for q in queues or []:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", q):
                raise ValueError("queues must be Yandex Tracker queue keys")
        app = ctx.request_context.lifespan_context
        issues_api = app.issues
        auth = get_yandex_auth(ctx)
        if queues is not None:
            scope = list(queues)
        else:
            scope = [
                q.key
                for q in await app.queues.queues_list(auth=auth)
                if getattr(q, "key", None)
            ]
        per_queue: dict[str, dict[str, Any]] = {}
        for q in scope:
            per_queue[q] = {
                "issues_scanned": 0,
                "issues_with_testing_history": 0,
                "test_cycles_total": 0,
                "test_hours": 0.0,
                "rework_total": 0,
                "rework_hours": 0.0,
                "in_rework_now": 0,
            }
        issues: list[Any] = []
        issue_queues: list[str] = []
        complete = True
        collected = 0
        per_queue_cap = queues is not None
        for q in scope:
            filters: dict[str, object] = {"queue": q}
            if version_id is not None:
                filters["fixVersions"] = version_id
            limit = max_issues if per_queue_cap else max_issues - collected
            if limit <= 0:
                break
            batch, q_complete = await _drain_filtered(
                issues_api,
                auth,
                filters,
                ["key", "summary", "status", "updatedAt"],
                limit,
                "testing_cycle",
            )
            complete = complete and q_complete
            per_queue[q]["issues_scanned"] = len(batch)
            issues.extend(batch)
            issue_queues.extend(q for _ in batch)
            collected += len(batch)
            if not per_queue_cap and collected >= max_issues:
                break
        test_cycles = 0
        test_hours = 0.0
        reworks = 0
        rework_hours = 0.0
        with_history = 0
        in_rework_now: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(4)

        async def fetch_changelog(issue: Any) -> list[Any]:
            async with semaphore:
                return await issues_api.issue_get_status_changelog(issue.key, auth=auth)

        changelogs = await asyncio.gather(*(fetch_changelog(issue) for issue in issues))
        for issue, queue_name, changelog in zip(
            issues, issue_queues, changelogs, strict=True
        ):
            events = _status_events(changelog)
            if not events:
                continue
            with_history += 1
            metrics = _cycle_metrics(events)
            test_cycles += metrics["test_cycles"]
            test_hours += metrics["test_hours_total"]
            reworks += metrics["rework_count"]
            rework_hours += metrics["rework_hours_total"]
            pq = per_queue[queue_name]
            pq["issues_with_testing_history"] += 1
            pq["test_cycles_total"] += metrics["test_cycles"]
            pq["test_hours"] += metrics["test_hours_total"]
            pq["rework_total"] += metrics["rework_count"]
            pq["rework_hours"] += metrics["rework_hours_total"]
            status_display = (
                (issue.status.display or "") if issue.status else ""
            ).casefold()
            if metrics["currently_in_rework"] and status_display == "в работе":
                pq["in_rework_now"] += 1
                in_rework_now.append(
                    {
                        "key": issue.key,
                        "summary": issue.summary,
                        "queue": queue_name,
                        "updated_at": (
                            issue.updated_at.isoformat()
                            if hasattr(issue.updated_at, "isoformat")
                            else None
                        ),
                        "url": f"https://tracker.yandex.ru/{issue.key}",
                    }
                )
        for q in scope:
            pq = per_queue[q]
            pq["avg_test_cycle_hours"] = (
                round(pq["test_hours"] / pq["test_cycles_total"], 2)
                if pq["test_cycles_total"]
                else None
            )
            pq["avg_rework_hours"] = (
                round(pq["rework_hours"] / pq["rework_total"], 2)
                if pq["rework_total"]
                else None
            )
            pq.pop("test_hours", None)
            pq.pop("rework_hours", None)
        in_rework_now.sort(key=lambda row: str(row["updated_at"] or ""), reverse=True)
        response = {
            "status": "complete",
            "complete": True,
            "queues": scope,
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
            "per_queue": per_queue,
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
            "QA-lead defect metrics, ORG-WIDE by default: for ALL queues "
            "(or the queues given) issues of type bug created in a date "
            "range — created/closed/open totals with a per-queue breakdown, "
            "weekly trend, open-bug aging buckets (<30 / 30-90 / >90 days), "
            "and escapes (issues whose tags or resolution contain "
            "escape_marker, e.g. «прод»). Use for «тренд багов», «старение "
            "багов», «утечки в прод» across the department. Closed dates are "
            "approximated by updated_at (reporting_contract). The escape "
            "table is capped (rows_capped in coverage)."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_defect_trend(
        ctx: Context[Any, AppContext],
        queues: Annotated[
            list[str] | None,
            Field(
                max_length=10,
                description="Queue keys; default: ALL queues (org-wide)",
            ),
        ] = None,
        created_after: Annotated[
            str, Field(description="ISO date, inclusive start of the range")
        ] = ...,
        created_before: Annotated[
            str | None,
            Field(description="ISO date, inclusive end; defaults to today"),
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
        app = ctx.request_context.lifespan_context
        issues_api = app.issues
        auth = get_yandex_auth(ctx)
        scope = queues or [
            q.key
            for q in await app.queues.queues_list(auth=auth)
            if getattr(q, "key", None)
        ]
        for queue in scope:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
                raise ValueError("queue must be a Yandex Tracker queue key")
        end = created_before or _now().date().isoformat()
        today = _now().date()

        created_total = 0
        closed_total = 0
        open_now = 0
        aging: Counter[str] = Counter()
        escapes: list[dict[str, Any]] = []
        trend: Counter[str] = Counter()
        per_queue: dict[str, dict[str, Any]] = {}
        processed = 0
        complete = True
        for queue in scope:
            issues, q_complete = await _drain_filtered(
                issues_api,
                auth,
                {
                    "queue": queue,
                    "type": bug_type_keys,
                    "created": {"from": created_after, "to": end},
                },
                [
                    "key",
                    "summary",
                    "queue",
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
            complete = complete and q_complete
            processed += len(issues)
            q_created = 0
            q_closed = 0
            q_open = 0
            for issue in issues:
                st = _reference_key(getattr(issue, "statusType", None))
                is_open = st not in _FINAL_STATUS_TYPE_KEYS
                q_created += 1
                if is_open:
                    q_open += 1
                else:
                    q_closed += 1
                if isinstance(issue.created_at, datetime):
                    trend[_bucket_label(issue.created_at, group_by)] += 1
                if is_open and isinstance(issue.created_at, datetime):
                    aging[_age_bucket((today - issue.created_at.date()).days)] += 1
                if escape_marker:
                    marker = escape_marker.casefold()
                    resolution = (
                        _reference_display(getattr(issue, "resolution", None)) or ""
                    )
                    if (
                        marker
                        in (" ".join(issue.tags or []) + " " + resolution).casefold()
                    ):
                        escapes.append(
                            {
                                "key": issue.key,
                                "summary": issue.summary,
                                "queue": _reference_display(
                                    getattr(issue, "queue", None)
                                )
                                or getattr(issue, "queue", None),
                                "status": issue.status.display
                                if issue.status
                                else None,
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
            created_total += q_created
            closed_total += q_closed
            open_now += q_open
            per_queue[queue] = {
                "created_total": q_created,
                "closed_total": q_closed,
                "open_now": q_open,
            }
        escapes.sort(key=lambda row: str(row["updated_at"] or ""), reverse=True)
        response = {
            "status": "complete",
            "complete": True,
            "queues": scope,
            "range": {"created_after": created_after, "created_before": end},
            "bug_type_keys": bug_type_keys,
            "created_total": created_total,
            "closed_total": closed_total,
            "open_now": open_now,
            "per_queue": per_queue,
            "aging_buckets_days": dict(aging),
            "trend": [
                {"period": bucket, "created": trend[bucket]} for bucket in sorted(trend)
            ],
            "coverage": {
                "processed_issues": processed,
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
        if escape_marker:
            response["escapes"] = len(escapes)
            response["escapes_share"] = (
                round(len(escapes) / created_total, 3) if created_total else None
            )
            response["escape_table"] = escapes
            return _cap_rows_for_budget(response, "escape_table")
        return response

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
        queues: Annotated[
            list[str] | None,
            Field(
                max_length=10,
                description="Queue keys; default: ALL queues with per-queue breakdown and org-wide totals",
            ),
        ] = None,
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
        for q in queues or []:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", q):
                raise ValueError("queues must be Yandex Tracker queue keys")
        app = ctx.request_context.lifespan_context
        issues_api = app.issues
        auth = get_yandex_auth(ctx)
        if queues is not None:
            scope = list(queues)
        else:
            scope = [
                q.key
                for q in await app.queues.queues_list(auth=auth)
                if getattr(q, "key", None)
            ]
        stale_cutoff = _now() - timedelta(days=stale_days)
        stale: list[dict[str, Any]] = []
        per_queue: dict[str, dict[str, Any]] = {}
        totals = {
            "issues_total": 0,
            "no_estimation": 0,
            "no_spent": 0,
            "no_description": 0,
            "stale_non_final": 0,
        }
        complete = True
        for q in scope:
            issues, q_complete = await _drain_filtered(
                issues_api,
                auth,
                {"queue": q},
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
            complete = complete and q_complete
            no_estimation = 0
            no_spent = 0
            no_description = 0
            q_stale = 0
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
                            q_stale += 1
                            stale.append(
                                {
                                    "key": issue.key,
                                    "summary": issue.summary,
                                    "queue": q,
                                    "updated_at": updated.isoformat(),
                                    "url": f"https://tracker.yandex.ru/{issue.key}",
                                }
                            )
                    except (ValueError, TypeError):
                        pass
            total = len(issues)
            totals["issues_total"] += total
            totals["no_estimation"] += no_estimation
            totals["no_spent"] += no_spent
            totals["no_description"] += no_description
            totals["stale_non_final"] += q_stale
            per_queue[q] = {
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
                "stale_non_final": q_stale,
            }
        stale.sort(key=lambda row: str(row["updated_at"] or ""))
        grand_total = totals["issues_total"]
        response = {
            "status": "complete",
            "complete": True,
            "queues": scope,
            "stale_days": stale_days,
            "totals": {
                "issues_total": grand_total,
                "without_estimation": {
                    "count": totals["no_estimation"],
                    "share": round(totals["no_estimation"] / grand_total, 3)
                    if grand_total
                    else None,
                },
                "without_spent": {
                    "count": totals["no_spent"],
                    "share": round(totals["no_spent"] / grand_total, 3)
                    if grand_total
                    else None,
                },
                "without_description": {
                    "count": totals["no_description"],
                    "share": round(totals["no_description"] / grand_total, 3)
                    if grand_total
                    else None,
                },
                "stale_non_final": totals["stale_non_final"],
            },
            "per_queue": per_queue,
            "stale_table": stale,
            "coverage": {
                "processed_issues": grand_total,
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
            str | None,
            Field(
                description="Queue key used to find the board; optional when board_id is given"
            ),
        ] = None,
        board_id: Annotated[
            int | None,
            Field(
                gt=0, description="Explicit board id; otherwise resolved by queue name"
            ),
        ] = None,
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if queue is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        if board_id is None:
            boards = await issues_api.boards_get_all(auth=auth)
            if queue is None:
                return {
                    "status": "board_not_found",
                    "complete": False,
                    "queue": None,
                    "candidate_boards": [],
                    "all_boards": [
                        {"id": board.get("id"), "name": board.get("name")}
                        for board in boards
                    ],
                    "required_action": "Supply board_id; do not report a carryover count.",
                }
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
                    "all_boards": [
                        {"id": board.get("id"), "name": board.get("name")}
                        for board in boards
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

    @mcp.tool(
        title="QA Department Dashboard",
        description=(
            "Weekly QA-department dashboard («дашборд отдела QA»): ONE call "
            "returns the org-wide QA load — in-testing and stale issues per "
            "queue across ALL queues (or the queues given) — plus release "
            "readiness and return rate for the active release (auto-resolved "
            "to the most loaded un-released version), testing-cycle averages "
            "from status history (sampled), defect trend and escapes for the "
            "release queue, and data discipline for the release and QA-team "
            "queues. Use whenever the user asks for «дашборд отдела QA», the "
            "QA department dashboard, or a weekly QA summary. Aggregates "
            "only — no per-task tables; point the user to the specific "
            "metric tools for detail."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_qa_dashboard(
        ctx: Context[Any, AppContext],
        queues: Annotated[
            list[str] | None,
            Field(
                max_length=10,
                description=(
                    "Queue keys for the workset scope; default: ALL queues. "
                    "Release/discipline always use YOURQUEUE+TESTQUEUE "
                    "unless queues is given (then its first queue anchors the "
                    "release)."
                ),
            ),
        ] = None,
        version_id: Annotated[
            int | None,
            Field(
                gt=0,
                description="Release version id; default: auto-resolve the most loaded un-released version",
            ),
        ] = None,
        created_after: Annotated[
            str | None,
            Field(
                description="ISO date, inclusive start of the defect window (default: 7 days ago)"
            ),
        ] = None,
        escape_marker: Annotated[
            str | None,
            Field(
                description="Substring matched against tags/resolution to detect escapes (e.g. «прод»)"
            ),
        ] = None,
        sample: Annotated[
            int,
            Field(
                ge=10,
                le=500,
                description="Max issues for changelog-based blocks (returns/cycle)",
            ),
        ] = 100,
        max_issues: Annotated[int, Field(ge=100, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        release_scope = queues or ["YOURQUEUE", "TESTQUEUE"]
        for queue in release_scope:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
                raise ValueError("queue must be a Yandex Tracker queue key")
        app = ctx.request_context.lifespan_context
        issues_api = app.issues
        queues_api = app.queues
        auth = get_yandex_auth(ctx)
        now = _now()
        today = now.date().isoformat()
        since = created_after or (now - timedelta(days=7)).date().isoformat()

        release_queue: str | None = None
        release_version: int | None = None
        release_name: str | None = None

        def _version_value(version: Any, *names: str) -> Any:
            for name in names:
                value = (
                    version.get(name)
                    if isinstance(version, dict)
                    else getattr(version, name, None)
                )
                if value:
                    return value
            return None

        if version_id is not None:
            release_queue, release_version = release_scope[0], version_id
        else:
            for queue in release_scope:
                try:
                    versions = await queues_api.queues_get_versions(queue, auth=auth)
                except Exception:  # noqa: BLE001 — resolution is best-effort
                    versions = []
                dated = [
                    version
                    for version in versions
                    if _version_value(
                        version, "dueDate", "due_date", "startDate", "start_date"
                    )
                ]
                if dated:
                    dated.sort(
                        key=lambda version: (
                            str(
                                _version_value(
                                    version,
                                    "dueDate",
                                    "due_date",
                                    "startDate",
                                    "start_date",
                                )
                                or ""
                            ),
                            int(_version_value(version, "id") or 0),
                        ),
                        reverse=True,
                    )
                    # Active release = the most loaded among the latest
                    # un-released dated versions (a brand-new version with 2
                    # tasks is not the one being tested; the loaded one is).
                    active = [
                        version
                        for version in dated
                        if not _version_value(version, "released")
                        and not _version_value(version, "archived")
                    ]
                    best: tuple[Any, int] | None = None
                    for candidate in (active or dated)[:10]:
                        candidate_issues, _ = await _drain_filtered(
                            issues_api,
                            auth,
                            {
                                "queue": queue,
                                "fixVersions": int(_version_value(candidate, "id")),
                            },
                            ["key"],
                            max_issues,
                            "qa_dashboard.version_count",
                        )
                        if best is None or len(candidate_issues) > best[1]:
                            best = (candidate, len(candidate_issues))
                    assert best is not None
                    release_queue, release_version = (
                        queue,
                        int(_version_value(best[0], "id")),
                    )
                    release_name = _version_value(best[0], "name")
                    break

        release: dict[str, Any] | None = None
        cycle: dict[str, Any] | None = None
        if release_queue is not None and release_version is not None:
            release_issues, release_complete = await _drain_filtered(
                issues_api,
                auth,
                {"queue": release_queue, "fixVersions": release_version},
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
                "qa_dashboard.release",
            )
            status_types: Counter[str] = Counter()
            open_critical: list[dict[str, Any]] = []
            with_estimation = 0
            bugs = 0
            for issue in release_issues:
                st = _reference_key(getattr(issue, "statusType", None))
                status_types[st or "unknown"] += 1
                is_open = st not in _FINAL_STATUS_TYPE_KEYS
                if issue.estimation:
                    with_estimation += 1
                if (_reference_key(getattr(issue, "type", None)) or "").casefold() in (
                    "bug",
                    "defect",
                ):
                    bugs += 1
                if (
                    is_open
                    and (
                        _reference_key(getattr(issue, "priority", None)) or ""
                    ).casefold()
                    in _CRITICAL_PRIORITIES
                ):
                    open_critical.append(
                        {
                            "key": issue.key,
                            "summary": issue.summary,
                            "priority": _reference_display(
                                getattr(issue, "priority", None)
                            ),
                        }
                    )
            open_critical.sort(key=lambda row: str(row["key"] or ""))
            total = len(release_issues)
            release = {
                "queue": release_queue,
                "version_id": release_version,
                "version_name": release_name,
                "issues_total": total,
                "status_type_counts": dict(status_types),
                "open_critical": len(open_critical),
                "open_critical_top": open_critical[:5],
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
                "coverage": {"complete": release_complete},
            }

            sampled_issues, _ = await _drain_filtered(
                issues_api,
                auth,
                {"queue": release_queue, "fixVersions": release_version},
                ["key", "summary", "status", "updatedAt"],
                sample,
                "qa_dashboard.cycle",
            )
            test_cycles = 0
            test_hours = 0.0
            reworks = 0
            rework_hours = 0.0
            issues_with_returns = 0
            semaphore = asyncio.Semaphore(4)

            async def fetch_changelog(issue: Any) -> list[Any]:
                async with semaphore:
                    return await issues_api.issue_get_status_changelog(
                        issue.key, auth=auth
                    )

            changelogs = await asyncio.gather(
                *(fetch_changelog(issue) for issue in sampled_issues)
            )
            for _issue, changelog in zip(sampled_issues, changelogs, strict=True):
                events = _status_events(changelog)
                if not events:
                    continue
                metrics = _cycle_metrics(events)
                test_cycles += metrics["test_cycles"]
                test_hours += metrics["test_hours_total"]
                reworks += metrics["rework_count"]
                rework_hours += metrics["rework_hours_total"]
                if metrics["rework_count"]:
                    issues_with_returns += 1
            cycle = {
                "queue": release_queue,
                "version_id": release_version,
                "issues_sampled": len(sampled_issues),
                "issues_with_returns": issues_with_returns,
                "return_rate_share": round(issues_with_returns / len(sampled_issues), 3)
                if sampled_issues
                else None,
                "rework_total": reworks,
                "test_cycles_total": test_cycles,
                "avg_test_cycle_hours": round(test_hours / test_cycles, 2)
                if test_cycles
                else None,
                "avg_rework_hours": round(rework_hours / reworks, 2)
                if reworks
                else None,
                "coverage": {"sampled": True, "sample_limit": sample},
            }

        defect_queue = release_queue or release_scope[0]
        defects, defect_complete = await _drain_filtered(
            issues_api,
            auth,
            {
                "queue": defect_queue,
                "type": ["bug"],
                "created": {"from": since, "to": today},
            },
            [
                "key",
                "status",
                "statusType",
                "priority",
                "resolution",
                "tags",
                "createdAt",
                "updatedAt",
            ],
            max_issues,
            "qa_dashboard.defects",
        )
        closed = 0
        open_now = 0
        aging: Counter[str] = Counter()
        escapes: list[str] = []
        trend: Counter[str] = Counter()
        for issue in defects:
            st = _reference_key(getattr(issue, "statusType", None))
            is_open = st not in _FINAL_STATUS_TYPE_KEYS
            if is_open:
                open_now += 1
            else:
                closed += 1
            if isinstance(issue.created_at, datetime):
                trend[_bucket_label(issue.created_at, "week")] += 1
            if is_open and isinstance(issue.created_at, datetime):
                aging[_age_bucket((now.date() - issue.created_at.date()).days)] += 1
            if escape_marker:
                marker = escape_marker.casefold()
                resolution = (
                    _reference_display(getattr(issue, "resolution", None)) or ""
                )
                if marker in (" ".join(issue.tags or []) + " " + resolution).casefold():
                    escapes.append(issue.key)
        defects_block: dict[str, Any] = {
            "queue": defect_queue,
            "created_total": len(defects),
            "closed_total": closed,
            "open_now": open_now,
            "aging_buckets_days": dict(aging),
            "trend_weekly": [
                {"period": bucket, "created": trend[bucket]} for bucket in sorted(trend)
            ],
            "coverage": {
                "complete": defect_complete,
                "window": {"from": since, "to": today},
            },
        }
        if escape_marker:
            defects_block["escapes"] = len(escapes)
            defects_block["escapes_share"] = (
                round(len(escapes) / len(defects), 3) if defects else None
            )
            defects_block["escape_keys"] = escapes[:5]

        discipline_per_queue: dict[str, dict[str, Any]] = {}
        for queue in release_scope:
            discipline, discipline_complete = await _drain_filtered(
                issues_api,
                auth,
                {"queue": queue},
                [
                    "key",
                    "status",
                    "statusType",
                    "estimation",
                    "spent",
                    "description",
                    "updatedAt",
                ],
                max_issues,
                "qa_dashboard.discipline",
            )
            no_estimation = 0
            no_spent = 0
            no_description = 0
            stale_non_final = 0
            stale_cutoff = now - timedelta(days=30)
            for issue in discipline:
                if not issue.estimation:
                    no_estimation += 1
                if not issue.spent:
                    no_spent += 1
                if not (issue.description or "").strip():
                    no_description += 1
                st = _reference_key(getattr(issue, "statusType", None))
                if st not in _FINAL_STATUS_TYPE_KEYS:
                    updated = issue.updated_at
                    if updated is not None:
                        try:
                            if updated.tzinfo is None:
                                updated = updated.replace(tzinfo=timezone.utc)
                            if updated < stale_cutoff:
                                stale_non_final += 1
                        except (ValueError, TypeError):
                            pass
            total = len(discipline)
            discipline_per_queue[queue] = {
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
                "stale_non_final_30d": stale_non_final,
                "coverage": {"complete": discipline_complete},
            }

        # Org-wide QA workset: in-testing issues per queue, status-filtered
        # server-side (cheap — only the testing-status rows are fetched).
        workset_queues = queues or [
            q.key
            for q in await queues_api.queues_list(auth=auth)
            if getattr(q, "key", None)
        ]
        statuses = await app.fields.get_statuses(auth=auth)
        testing_keys: list[str] = []
        for status in statuses:
            display = " ".join(
                (
                    getattr(status, "name", None)
                    or getattr(status, "display", None)
                    or ""
                ).split()
            )
            if display.casefold() in _TESTING_DISPLAYS:
                key = getattr(status, "key", None)
                if key:
                    testing_keys.append(key)
        workset_per_queue: dict[str, dict[str, Any]] = {}
        total_in_testing = 0
        total_stale = 0
        workset_complete = True
        stale_ws_cutoff = now - timedelta(days=5)
        for queue in workset_queues:
            if not testing_keys:
                workset_per_queue[queue] = {
                    "in_testing": 0,
                    "stale_in_testing_5d": 0,
                }
                workset_complete = False
                continue
            ws_issues, ws_complete = await _drain_filtered(
                issues_api,
                auth,
                {"queue": queue, "status": testing_keys},
                ["key", "updatedAt"],
                max_issues,
                "qa_dashboard.workset",
            )
            workset_complete = workset_complete and ws_complete
            stale = 0
            for issue in ws_issues:
                updated = issue.updated_at
                if updated is not None:
                    try:
                        if updated.tzinfo is None:
                            updated = updated.replace(tzinfo=timezone.utc)
                        if updated < stale_ws_cutoff:
                            stale += 1
                    except (ValueError, TypeError):
                        pass
            workset_per_queue[queue] = {
                "in_testing": len(ws_issues),
                "stale_in_testing_5d": stale,
            }
            total_in_testing += len(ws_issues)
            total_stale += stale

        response = {
            "status": "complete",
            "complete": True,
            "workset_queues": workset_queues,
            "as_of": today,
            "release": release,
            "cycle": cycle,
            "workset": {
                "total_in_testing": total_in_testing,
                "total_stale_in_testing_5d": total_stale,
                "stale_days": 5,
                "per_queue": workset_per_queue,
                "coverage": {
                    "complete": workset_complete,
                    "testing_statuses_resolved": bool(testing_keys),
                },
            },
            "defects": defects_block,
            "discipline": discipline_per_queue,
            "reporting_contract": {
                "aggregates_only_no_tables": True,
                "cycle_and_returns_are_sampled": True,
                "hours_not_days": True,
                "workset_scope_is_all_queues_by_default": True,
                "release_anchor_is_most_loaded_unreleased_version": True,
            },
        }
        return response
