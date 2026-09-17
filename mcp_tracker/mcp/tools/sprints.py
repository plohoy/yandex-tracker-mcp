"""Board-sprint tools for Yandex Tracker: any-depth sprint catalogue, a single
sprint's results (plan/fact hours, per-assignee split, per-issue table) and
multi-sprint history analytics.

Tracker API facts these tools rely on (probed live against the org API):

- ``GET v3/sprints/{id}`` resolves a sprint at ANY depth and returns its board,
  so a bare sprint id (e.g. from ``https://tracker.yandex.ru/issues/?sprint=552``)
  needs no board lookup at all;
- ``GET v3/boards/{id}/sprints`` lists every sprint of one board; the ``board``
  query parameter of ``GET v3/sprints`` is ignored server-side, so an org-wide
  sprint list cannot be filtered by board;
- ``POST v3/issues/_search`` honours the structured ``sprint: [id]`` filter only
  in the nested ``{"filter": {...}}`` body — exactly what
  ``TrackerClient.issues_find_filter`` sends. A BARE ``{"sprint": [id]}`` body is
  silently ignored and returns the whole org, so the sprint scope must always go
  through the client method.

Durations follow the fork convention: 8 hours per day, 5 days per week.
"""

import asyncio
import re
from collections import Counter
from datetime import date
from typing import Annotated, Any, Literal

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_tracker.mcp.context import AppContext
from mcp_tracker.mcp.tools.issue_read import (
    _FINAL_STATUS_TYPE_KEYS,
    _cap_rows_for_budget,
    _count_issue_returns,
    _duration_hours,
    _reference_display,
    _reference_key,
)
from mcp_tracker.mcp.tools.metrics import _drain_filtered
from mcp_tracker.mcp.utils import get_yandex_auth
from mcp_tracker.settings import Settings
from mcp_tracker.tracker.custom.errors import SprintNotFound

_SPRINT_FIELDS = [
    "key",
    "summary",
    "status",
    "statusType",
    "assignee",
    "queue",
    "estimation",
    "spent",
]

# Rows in the pre-built per-issue table (the model copies it verbatim; the
# structured `rows` list is capped separately by the response budget). The
# 8-column table with verbatim summaries also carries its own char budget.
_TABLE_ROW_LIMIT = 150
_TABLE_BUDGET_CHARS = 12_000
_NO_ASSIGNEE = "(без исполнителя)"
_DASH = "—"
# Per-sprint states that carry no figures and must not count as analysed.
_SPRINT_FAILURE_STATES = frozenset({"sprint_error", "sprint_not_found"})
# Changelog fetches run with bounded concurrency (status history per issue).
_RETURNS_CONCURRENCY = 4
_RETURN_METRICS = ("qa_rework_cycle", "testing_rework", "repeated_work_status")


def _hours_or_none(value: object) -> float | None:
    """Duration -> hours, None when empty, 0.0 when unparseable (fail-soft)."""
    if value in (None, ""):
        return None
    try:
        return _duration_hours(value)
    except ValueError:
        return 0.0


def _fmt_hours(value: float | None) -> str:
    if value is None:
        return _DASH
    return f"{round(value, 2):g}"


def _fmt_returns(value: object) -> str:
    """Return count cell: a number, or — when the issue was not scanned."""
    if value is None:
        return _DASH
    return str(value)


def _queue_stem(queue: str) -> str:
    """Queue key -> board-name stem for the fallback board search.

    Boards are often named after the product, not after the queue key
    (a queue key YOURQUEUE may live on a board called «Your Product Sprint»),
    so the suffix is dropped before matching board names.
    """
    key = queue.casefold()
    for suffix in ("team", "squad", "dept"):
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)]
    return key


def _stem_candidates(boards: list[dict[str, Any]], queue: str) -> list[dict[str, Any]]:
    """Boards whose name contains the queue stem (candidates for the user to pick)."""
    stem = _queue_stem(queue)
    out: list[dict[str, Any]] = []
    for board in boards:
        name = str(board.get("name", "")).casefold()
        if stem and stem in name:
            out.append({"id": board.get("id"), "name": board.get("name")})
    return out[:10]


def _board_name_matches(name: object, query: str) -> bool:
    """Case-insensitive partial match in BOTH directions.

    A sprint or board name in a user request is usually longer than the board
    name («Product QA Sprint 25» vs board «Product QA Sprint») and a queue key is
    usually shorter («YOURQUEUE» vs board «Product Sprint»), so a match counts when
    either normalized string contains the other.
    """
    board = " ".join(str(name or "").casefold().split())
    needle = " ".join(query.casefold().split())
    return bool(board) and bool(needle) and (needle in board or board in needle)


def _board_rows(boards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": board.get("id"), "name": board.get("name")} for board in boards]


async def _resolve_board(
    issues_api: Any,
    auth: Any,
    *,
    queue: str | None,
    board_name: str | None,
    board_id: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Resolve the sprint board from board_id, a board name or a queue key.

    Returns ``(board, error_payload, boards)``: on success ``error_payload`` is
    None; on failure ``board`` is None and the caller returns ``error_payload``
    verbatim. ``boards`` is the fetched board catalogue (None when board_id was
    supplied), reused for candidate lists by the callers.
    """
    if board_id is not None:
        return {"id": board_id, "name": None}, None, None
    boards = await issues_api.boards_get_all(auth=auth)
    all_boards = _board_rows(boards)
    if board_name:
        matches = [b for b in boards if _board_name_matches(b.get("name"), board_name)]
        if len(matches) == 1:
            return matches[0], None, boards
        return (
            None,
            {
                "status": "ambiguous_board" if matches else "board_not_found",
                "complete": False,
                "board_name": board_name,
                "candidate_boards": _board_rows(matches),
                "all_boards": all_boards,
                "required_action": (
                    "Pass board_id from candidate_boards, or pass sprint_id / "
                    "sprint_ids to address sprints directly. Do not guess a board."
                ),
            },
            boards,
        )
    if queue is None:
        return (
            None,
            {
                "status": "board_not_found",
                "complete": False,
                "queue": None,
                "board_name": None,
                "candidate_boards": [],
                "all_boards": all_boards,
                "required_action": (
                    "Name the queue, pass board_name (part of the board or "
                    "product name), or pass sprint_id to resolve one sprint "
                    "directly."
                ),
            },
            boards,
        )
    needle = f" {queue.casefold()} "
    candidates = [
        board
        for board in boards
        if needle in f" {str(board.get('name', '')).casefold()} "
    ]
    if len(candidates) == 1:
        return candidates[0], None, boards
    return (
        None,
        {
            "status": "ambiguous_board" if candidates else "board_not_found",
            "complete": False,
            "queue": queue,
            "candidate_boards": _board_rows(candidates),
            "stem_candidate_boards": _stem_candidates(boards, queue),
            "all_boards": all_boards,
            "required_action": (
                "Show candidate_boards and stem_candidate_boards and ask for "
                "board_id (org sprint boards are often named after the product, "
                "not the queue key), or pass sprint_id to resolve one sprint "
                "directly. Do not guess."
            ),
        },
        boards,
    )


def _sprint_row(sprint: dict[str, Any]) -> dict[str, Any]:
    """Normalise a Tracker sprint object for tool output."""
    board = sprint.get("board")
    board = board if isinstance(board, dict) else {}
    state = sprint.get("status") or ("archived" if sprint.get("archived") else None)
    return {
        "id": sprint.get("id"),
        "name": sprint.get("name"),
        "start_date": sprint.get("startDate") or sprint.get("start_date"),
        "end_date": sprint.get("endDate") or sprint.get("end_date"),
        "state": state,
        "board_id": board.get("id"),
        "board_name": board.get("display") or board.get("name"),
    }


def _is_current(sprint: dict[str, Any], today: str) -> bool:
    if sprint.get("state") == "in_progress":
        return True
    start = str(sprint.get("start_date") or "")
    end = str(sprint.get("end_date") or "")
    return bool(start and end and start <= today <= end)


def _as_int(value: object) -> int:
    """Best-effort int for Tracker's str-or-int ids (0 when unusable)."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _key_order(key: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*)-(\d+)", str(key))
    if not match:
        return (str(key), 0, str(key))
    return (match.group(1), int(match.group(2)), "")


def _sprint_table(rows: list[dict[str, Any]]) -> str:
    """Pre-built markdown table «Очередь | Номер Задачи | Статус | Исполнитель |
    План часов | Факт часов» (no code fence — the user's surface renders real
    tables). The model copies it verbatim; rows are sorted queue-less."""
    headers = [
        "Очередь",
        "Номер Задачи",
        "Заголовок",
        "Статус",
        "Исполнитель",
        "План часов",
        "Факт часов",
        "Количество возвратов",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("queue") or _DASH),
                    str(row.get("key") or _DASH),
                    str(row.get("summary") or _DASH),
                    str(row.get("status") or _DASH),
                    str(row.get("assignee") or _DASH),
                    _fmt_hours(row.get("plan_hours")),
                    _fmt_hours(row.get("fact_hours")),
                    _fmt_returns(row.get("returns")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _table_rows_for_budget(
    rows: list[dict[str, Any]], budget: int = _TABLE_BUDGET_CHARS
) -> list[dict[str, Any]]:
    """Longest prefix of rows whose rendered table stays inside the budget.

    Summaries are copied verbatim (never shortened), so a sprint with long
    titles needs fewer rows per answer; the remainder is reported in coverage
    instead of blowing the response budget.
    """
    if not rows or len(_sprint_table(rows)) <= budget:
        return rows
    low, high = 1, len(rows)
    while low < high:
        mid = (low + high + 1) // 2
        if len(_sprint_table(rows[:mid])) <= budget:
            low = mid
        else:
            high = mid - 1
    return rows[:low]


def _history_table(per_sprint: list[dict[str, Any]]) -> str:
    """Pre-built markdown table of per-sprint analytics (chronological)."""
    headers = [
        "Спринт",
        "Период",
        "Задач",
        "Выполнено",
        "Осталось",
        "План",
        "Факт",
        "Выполнено %",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for entry in per_sprint:
        period = (
            f"{entry.get('start_date') or _DASH} – {entry.get('end_date') or _DASH}"
        )
        share = entry.get("completion_share")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{entry.get('name') or _DASH} (#{entry.get('id')})",
                    period,
                    str(entry.get("issues_total", _DASH)),
                    str(entry.get("done", _DASH)),
                    str(entry.get("remaining", _DASH)),
                    _fmt_hours(entry.get("plan_hours")),
                    _fmt_hours(entry.get("fact_hours")),
                    _DASH if share is None else f"{round(share * 100)}%",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


async def _sprint_metrics(
    issues_api: Any,
    auth: Any,
    sprint_id: int,
    queue: str | None,
    max_issues: int,
    *,
    with_returns: bool = False,
    returns_metric: str = "qa_rework_cycle",
    max_return_scans: int = 300,
) -> dict[str, Any]:
    """Counts, plan/fact hours, per-assignee split and per-issue rows of one sprint.

    Single sprint scope, one drain attempt per page: the sprint filter is served
    through the client's nested ``{"filter": {"sprint": [id]}}`` body.
    """
    filters: dict[str, object] = {"sprint": [sprint_id]}
    if queue is not None:
        filters["queue"] = queue
    issues, complete = await _drain_filtered(
        issues_api,
        auth,
        filters,
        list(_SPRINT_FIELDS),
        max_issues,
        "sprint_results",
    )
    status_type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    queue_counts: Counter[str] = Counter()
    assignees: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    plan_hours = 0.0
    fact_hours = 0.0
    with_estimation = 0
    with_spent = 0
    done = 0
    unparsed = 0
    for issue in issues:
        status_type = _reference_key(getattr(issue, "statusType", None)) or "unknown"
        status_type_counts[status_type] += 1
        status_display = _reference_display(issue.status) or _DASH
        status_counts[status_display] += 1
        queue_key = _reference_key(getattr(issue, "queue", None)) or ""
        if queue_key:
            queue_counts[queue_key] += 1
        assignee = _reference_display(getattr(issue, "assignee", None)) or _NO_ASSIGNEE
        raw_est = getattr(issue, "estimation", None)
        raw_spent = getattr(issue, "spent", None)
        plan = _hours_or_none(raw_est)
        fact = _hours_or_none(raw_spent)
        for raw, parsed in ((raw_est, plan), (raw_spent, fact)):
            if raw not in (None, "") and parsed == 0.0:
                unparsed += 1
        if plan is not None:
            plan_hours += plan
            with_estimation += 1
        if fact is not None:
            fact_hours += fact
            with_spent += 1
        is_done = status_type in _FINAL_STATUS_TYPE_KEYS
        if is_done:
            done += 1
        entry = assignees.setdefault(
            assignee,
            {
                "assignee": assignee,
                "issues_total": 0,
                "done": 0,
                "remaining": 0,
                "plan_hours": 0.0,
                "fact_hours": 0.0,
            },
        )
        entry["issues_total"] += 1
        entry["done" if is_done else "remaining"] += 1
        entry["plan_hours"] += plan or 0.0
        entry["fact_hours"] += fact or 0.0
        rows.append(
            {
                "key": issue.key,
                "summary": issue.summary,
                "status": status_display,
                "status_type": status_type,
                "assignee": None if assignee == _NO_ASSIGNEE else assignee,
                "queue": queue_key,
                "plan_hours": plan,
                "fact_hours": fact,
                "returns": None,
                "url": f"https://tracker.yandex.ru/{issue.key}",
            }
        )
    rows.sort(key=lambda row: _key_order(row["key"]))
    returns_info: dict[str, Any] = {
        "returns_metric": returns_metric if with_returns else None,
        "returns_total": 0,
        "issues_with_returns": 0,
        "returns_scanned": 0,
        "returns_skipped": 0,
        "returns_failed": 0,
    }
    if with_returns and rows:
        # Status history is one request per issue: bounded concurrency, a hard
        # scan cap and per-issue failure isolation (a bad changelog must not
        # cost the whole report). Unscanned rows keep returns=None → "—" in
        # the table, never a silent zero.
        semaphore = asyncio.Semaphore(_RETURNS_CONCURRENCY)
        targets = rows[:max_return_scans]
        returns_info["returns_skipped"] = len(rows) - len(targets)

        async def _scan(row: dict[str, Any]) -> None:
            async with semaphore:
                try:
                    changes = await issues_api.issue_get_status_changelog(
                        row["key"], auth=auth
                    )
                except Exception:  # noqa: BLE001 — isolate one bad changelog
                    returns_info["returns_failed"] += 1
                    return
            count, _evidence, _transitions = _count_issue_returns(
                changes, returns_metric
            )
            row["returns"] = count
            returns_info["returns_scanned"] += 1
            returns_info["returns_total"] += count
            if count:
                returns_info["issues_with_returns"] += 1

        await asyncio.gather(*(_scan(row) for row in targets))
    breakdown = sorted(
        assignees.values(), key=lambda item: (-item["issues_total"], item["assignee"])
    )
    for entry in breakdown:
        entry["plan_hours"] = round(entry["plan_hours"], 2)
        entry["fact_hours"] = round(entry["fact_hours"], 2)
    issues_total = len(issues)
    return {
        "issues_total": issues_total,
        "drain_complete": complete,
        "done": done,
        "remaining": issues_total - done,
        "completion_share": round(done / issues_total, 3) if issues_total else None,
        "status_type_counts": dict(status_type_counts.most_common()),
        "status_counts": dict(status_counts.most_common()),
        "queue_counts": dict(queue_counts.most_common()),
        "plan_hours": round(plan_hours, 2),
        "fact_hours": round(fact_hours, 2),
        "issues_with_estimation": with_estimation,
        "issues_with_spent": with_spent,
        "issues_without_assignee": sum(1 for row in rows if row["assignee"] is None),
        "assignee_breakdown": breakdown,
        "rows": rows,
        "unparsed_duration_values": unparsed,
        "returns": returns_info,
    }


def register_sprint_tools(settings: Settings, mcp: FastMCP[Any]) -> None:
    """Register board-sprint tools (read-only)."""

    @mcp.tool(
        title="List Board Sprints or Resolve a Sprint by id",
        description=(
            "SPRINT CATALOGUE at any depth. Two shapes: "
            "(1) sprint_id given — resolves that exact sprint by id through "
            "GET v3/sprints/{id} (works for ANY sprint, current or years old, and "
            "returns its board, dates and state) — use it for a pasted link such as "
            "https://tracker.yandex.ru/issues/?sprint=<id>; "
            "(2) queue or board_id given — lists EVERY sprint of that board, "
            "newest first, each with id, name, start/end dates and state "
            "(draft / in_progress / archived), plus the id of the current sprint. "
            "Use it for «какие спринты есть в очереди X», «список спринтов доски», "
            "«какой спринт сейчас идёт», «найди спринт по номеру/дате», before "
            "When the user gives only a sprint or board NAME (for example «дай "
            "информацию по спринту Product QA Sprint 25»), pass board_name with "
            "that name — the tool matches it against board names in both "
            "directions and lists the board's sprints in ONE call; never invent a "
            "queue key out of a sprint name. "
            "calling issues_metrics_sprint_results or issues_metrics_sprint_history "
            "(they need sprint ids). Board resolution is by queue key matched "
            "against board names: board_not_found (with all_boards), "
            "ambiguous_board (with candidate_boards + stem_candidate_boards) and "
            "board_without_sprints (the matched board carries no sprints). Org "
            "sprint boards are often named after the PRODUCT, not the queue key "
            "(a queue key YOURQUEUE may live on «Your Product Sprint»), which is "
            "why those answers carry candidate lists and must be reported as "
            "such — never guess a board; "
            "when candidates are returned, ask the user for board_id or accept a "
            "sprint_id. Sprints are the board's own sprints; a sprint id from a "
            "link resolves at ANY depth. Future sprints are included unless "
            "include_future=false. "
            "This tool returns metadata only: one call, no loops, no browser, no "
            "terminal, no per-issue fetches."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_list_sprints(
        ctx: Context[Any, AppContext],
        queue: Annotated[
            str | None,
            Field(
                description=(
                    "Queue key used to find the board (for example YOURQUEUE); "
                    "optional when board_id or sprint_id is given"
                )
            ),
        ] = None,
        board_id: Annotated[
            int | None,
            Field(
                gt=0,
                description="Explicit board id; otherwise resolved by board_name or queue",
            ),
        ] = None,
        board_name: Annotated[
            str | None,
            Field(
                max_length=128,
                description=(
                    "Part of the board or product name as the user wrote it "
                    "(for example a sprint title such as 'Product QA Sprint 25' "
                    "matches the board 'Product QA Sprint'); matched in both "
                    "directions, case-insensitively. Use it instead of inventing "
                    "a queue key when the user names a sprint/board only."
                ),
            ),
        ] = None,
        sprint_id: Annotated[
            int | None,
            Field(
                gt=0,
                description=(
                    "Resolve ONE sprint by id instead of listing a board's sprints"
                ),
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=200,
                description="Max sprints returned in the board listing (newest first)",
            ),
        ] = 30,
        include_future: Annotated[
            bool,
            Field(description="Include sprints that start after today"),
        ] = True,
    ) -> dict[str, Any]:
        app = ctx.request_context.lifespan_context
        issues_api = app.issues
        auth = get_yandex_auth(ctx)
        today = date.today().isoformat()
        if sprint_id is not None:
            try:
                sprint = await issues_api.sprint_get(sprint_id, auth=auth)
            except SprintNotFound:
                return {
                    "status": "sprint_not_found",
                    "complete": True,
                    "sprint_id": sprint_id,
                    "required_action": (
                        "Report that the sprint id does not exist; do not search "
                        "boards or issues to guess it."
                    ),
                }
            except Exception as exc:  # noqa: BLE001 — surface upstream failure
                return {
                    "status": "upstream_error",
                    "complete": False,
                    "sprint_id": sprint_id,
                    "error": type(exc).__name__,
                    "required_action": "Retry the same call once.",
                }
            row = _sprint_row(sprint)
            return {
                "status": "complete",
                "complete": True,
                "mode": "sprint_id",
                "sprint": row,
                "is_current": _is_current(row, today),
                "next_step": (
                    "Call issues_metrics_sprint_results(sprint_id=...) for the "
                    "figures of this sprint."
                ),
            }
        if queue is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        board, error, boards = await _resolve_board(
            issues_api,
            auth,
            queue=queue,
            board_name=board_name,
            board_id=board_id,
        )
        if error is not None:
            return error
        if board is None:  # pragma: no cover - helper contract
            return {"status": "board_not_found", "complete": False}
        try:
            sprints = await issues_api.board_get_sprints(
                _as_int(board["id"]), auth=auth
            )
        except Exception as exc:  # noqa: BLE001 — boards without sprints answer 400
            sprints = []
            sprints_error = type(exc).__name__
        else:
            sprints_error = None
        if not sprints:
            # Org sprint boards are named after the product, not the queue key;
            # offer stem-matched candidates instead of failing silently.
            candidates = _stem_candidates(boards or [], queue) if queue else []
            return {
                "status": "board_without_sprints",
                "complete": False,
                "board_id": board["id"],
                "board_name": board.get("name"),
                "error": sprints_error,
                "candidate_boards": candidates,
                "candidate_basis": "board names matching the queue product stem",
                "required_action": (
                    "This board carries no sprints; pass the board_id of the "
                    "right sprint board from candidate_boards, or pass sprint_id "
                    "to resolve one sprint directly. Do not guess a board."
                ),
            }
        rows = [_sprint_row(sprint) for sprint in sprints]
        if not include_future:
            rows = [row for row in rows if str(row.get("start_date") or "") <= today]
        rows.sort(
            key=lambda row: (
                str(row.get("end_date") or row.get("start_date") or ""),
                _as_int(row.get("id")),
            ),
            reverse=True,
        )
        total = len(rows)
        returned = rows[:limit]
        current = next((row for row in returned if _is_current(row, today)), None)
        return {
            "status": "complete",
            "complete": True,
            "mode": "board",
            "board_id": board["id"],
            "board_name": board.get("name"),
            "sprints": returned,
            "counts": {
                "sprints_total": total,
                "sprints_returned": len(returned),
                "limit": limit,
                "include_future": include_future,
            },
            "current_sprint": current,
            "coverage": {
                "complete": True,
                "sprints_total": total,
                "rows_capped": total > len(returned),
                "rows_total_sprints": total,
                "rows_returned_sprints": len(returned),
            },
            "reporting_contract": {
                "newest_first": True,
                "states": "draft / in_progress / archived",
                "sprint_ids_feed": (
                    "issues_metrics_sprint_results / issues_metrics_sprint_history"
                ),
            },
        }

    @mcp.tool(
        title="Sprint Results: Plan vs Fact Hours and Per-Person Split",
        description=(
            "ONE-SHOT results report for ONE board sprint at any depth, with the "
            "per-issue return count column: use for "
            "«результаты спринта», «собери результаты прошедшего спринта», "
            "«сколько задач выполнено и сколько осталось», «план/факт по часам», "
            "«по исполнителям», «разбивка спринта по статусам», or a pasted "
            "https://tracker.yandex.ru/issues/?sprint=<id> link. Pass sprint_id "
            "(from the link or from issues_list_sprints); queue is an optional "
            "narrowing filter, NOT required. Returns complete counters: issues "
            "total, done (statusType done/cancelled), remaining, per-status and "
            "per-status-type distributions, per-queue split, plan hours (sum of "
            "estimation), fact hours (sum of spent), and a per-assignee "
            "breakdown. A PRE-BUILT markdown table (key 'table') in columns "
            "«Очередь | Номер Задачи | Заголовок | Статус | Исполнитель | План "
            "часов | Факт часов | Количество возвратов» must be copied VERBATIM "
            "with every "
            "row on its own line — never rebuild, shorten or re-order it. The "
            "returns column counts each issue's status history with the same "
            "metrics as issues_count_release_status_returns (default "
            "qa_rework_cycle — one complete Тестируется→Провал→В работе→…→"
            "Тестируется loop), one changelog request per issue, 4 in flight, "
            "bounded by max_return_scans; '—' means NOT scanned (cap or a failed "
            "changelog) and must never be reported as zero returns — quote "
            "counts.returns_total with coverage.returns_scanned/returns_skipped. "
            "Hours are 8h/day, 5d/week; "
            "fact = logged spent time (not task lifetime) — say so. This is NOT "
            "a release-version tool: for version_id use "
            "issues_metrics_release_readiness / issues_summarize_effort. Never "
            "follow it with issue_get, worklogs, terminal, browser, tool search "
            "or per-issue loops; transient API errors are retried inside the "
            "tool, so retry the SAME call once instead of assembling the report "
            "from other tools."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_sprint_results(
        ctx: Context[Any, AppContext],
        sprint_id: Annotated[
            int,
            Field(
                gt=0,
                description="Board sprint id (from a link or from issues_list_sprints)",
            ),
        ],
        queue: Annotated[
            str | None,
            Field(
                description=(
                    "Optional queue key to narrow the sprint scope (for example "
                    "YOURQUEUE); omit for the whole sprint"
                )
            ),
        ] = None,
        include_table: Annotated[
            bool,
            Field(
                description=(
                    "Return the pre-built per-issue markdown table (key 'table'). "
                    "Set false when only counters are requested."
                )
            ),
        ] = True,
        max_issues: Annotated[
            int,
            Field(
                ge=1,
                le=10_000,
                description=(
                    "Safety ceiling for the sprint scan; the tool reports "
                    "coverage.complete=false when the sprint is larger"
                ),
            ),
        ] = 2_000,
        returns_metric: Annotated[
            Literal["qa_rework_cycle", "testing_rework", "repeated_work_status"],
            Field(
                description=(
                    "How the «Количество возвратов» column is counted (same "
                    "semantics as issues_count_release_status_returns): "
                    "qa_rework_cycle = one complete Тестируется→Провал→В работе→"
                    "…→Тестируется loop (default), testing_rework = the exact "
                    "Тестируется→Провал / Можно тестировать→Ревью transitions, "
                    "repeated_work_status = every visit after the first to a "
                    "working status"
                )
            ),
        ] = "qa_rework_cycle",
        max_return_scans: Annotated[
            int,
            Field(
                ge=0,
                le=2_000,
                description=(
                    "How many issues get a return count (one status-changelog "
                    "request each, 4 in flight). Rows beyond the cap show '—' in "
                    "the returns column and are counted in "
                    "coverage.returns_skipped — never report them as 0 returns. "
                    "0 skips the column entirely."
                ),
            ),
        ] = 300,
    ) -> dict[str, Any]:
        if queue is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        app = ctx.request_context.lifespan_context
        issues_api = app.issues
        auth = get_yandex_auth(ctx)
        try:
            sprint = await issues_api.sprint_get(sprint_id, auth=auth)
        except SprintNotFound:
            return {
                "status": "sprint_not_found",
                "complete": True,
                "sprint_id": sprint_id,
                "required_action": (
                    "Report that no such sprint exists; list the board's sprints "
                    "with issues_list_sprints instead of guessing."
                ),
            }
        sprint_row = _sprint_row(sprint)
        try:
            metrics = await _sprint_metrics(
                issues_api,
                auth,
                sprint_id,
                queue,
                max_issues,
                with_returns=max_return_scans > 0,
                returns_metric=returns_metric,
                max_return_scans=max_return_scans,
            )
        except Exception as exc:  # noqa: BLE001 — surface upstream failure
            return {
                "status": "upstream_error",
                "complete": False,
                "sprint_id": sprint_id,
                "sprint": sprint_row,
                "error": type(exc).__name__,
                "required_action": "Retry the same call once.",
            }
        rows = metrics["rows"]
        table_rows = _table_rows_for_budget(rows[:_TABLE_ROW_LIMIT])
        table_trimmed = len(table_rows) < len(rows)
        payload: dict[str, Any] = {
            "status": "complete",
            "complete": metrics["drain_complete"],
            "sprint": sprint_row,
            "filter": {"sprint_id": sprint_id, "queue": queue},
            "counts": {
                "issues_total": metrics["issues_total"],
                "done": metrics["done"],
                "remaining": metrics["remaining"],
                "completion_share": metrics["completion_share"],
                "status_type_counts": metrics["status_type_counts"],
                "status_counts": metrics["status_counts"],
                "queue_counts": metrics["queue_counts"],
                "issues_without_assignee": metrics["issues_without_assignee"],
                "returns_total": metrics["returns"]["returns_total"],
                "issues_with_returns": metrics["returns"]["issues_with_returns"],
                "returns_metric": metrics["returns"]["returns_metric"],
            },
            "hours": {
                "plan_hours": metrics["plan_hours"],
                "fact_hours": metrics["fact_hours"],
                "issues_with_estimation": metrics["issues_with_estimation"],
                "issues_with_spent": metrics["issues_with_spent"],
                "convention": "8h/day, 5d/week; fact = logged spent time",
            },
            "assignee_breakdown": metrics["assignee_breakdown"],
            "rows": rows,
            "coverage": {
                "complete": metrics["drain_complete"],
                "issues_scanned": metrics["issues_total"],
                "max_issues": max_issues,
                "table_rows_returned": len(table_rows),
                "table_rows_total": len(rows),
                "table_trimmed_by_size": table_trimmed,
                "table_budget_chars": _TABLE_BUDGET_CHARS,
                "unparsed_duration_values": metrics["unparsed_duration_values"],
                "returns_scanned": metrics["returns"]["returns_scanned"],
                "returns_skipped": metrics["returns"]["returns_skipped"],
                "returns_failed": metrics["returns"]["returns_failed"],
                "returns_note": (
                    "A '—' in the returns column means the issue was NOT scanned "
                    "(cap or changelog failure), not zero returns."
                ),
                "sprint_filter": "structured sprint filter via the client",
            },
            "reporting_contract": {
                "copy_table_verbatim": True,
                "done_means": "statusType in (done, cancelled)",
                "state_fact_meaning": True,
                "no_followup_tools": True,
            },
        }
        if include_table:
            payload["table"] = _sprint_table(table_rows)
            payload["table_note"] = (
                "Pre-built markdown table (columns Очередь | Номер Задачи | Заголовок "
                "| Статус | Исполнитель | План часов | Факт часов | Количество "
                "возвратов); copy it verbatim, every row on its own line, no code "
                "fence, and keep summaries complete — never shorten them. "
                + (
                    f"Rows shown: {len(table_rows)} of {len(rows)} "
                    "(the table is bounded to fit the response budget; say so and "
                    "quote coverage.table_rows_total)."
                    if table_trimmed
                    else "All rows of the scan are shown."
                )
            )
        if not metrics["drain_complete"]:
            payload["coverage"]["reason"] = (
                f"sprint holds more than max_issues={max_issues} issues; figures "
                "are a sample — state that explicitly"
            )
        return _cap_rows_for_budget(payload, "rows", "assignee_breakdown")

    @mcp.tool(
        title="Sprint History Analytics Across Several Sprints",
        description=(
            "MULTI-SPRINT analytics in ONE call: use for «аналитика по нескольким "
            "прошедшим спринтам», «сравни спринты», «динамика по спринтам», "
            "«сколько задач выполнялось в последних спринтах», «тренд "
            "выполнения/план-факт по спринтам очереди X». Scope: pass queue, "
            "board_name (part of the board/product name as the user wrote it) or "
            "board_id to take the board's last N started sprints (newest first, "
            "the in-progress one included unless include_current=false), or pass "
            "explicit sprint_ids for any depth/spread. Each sprint is scanned "
            "through the same single-sprint report core, and the response carries "
            "per-sprint figures (issues/done/remaining/plan hours/fact hours/"
            "completion share) plus aggregates (totals, average completion, plan "
            "vs fact across the window) and a PRE-BUILT markdown table (key "
            "'history_table') that must be copied VERBATIM, every row on its own "
            "line, no code fence. Per-sprint failures are isolated (status "
            "sprint_error on that row) — report them instead of dropping rows. "
            "Cost scales with last_n: each sprint is one bounded scan, so keep "
            "last_n small (default 5) and max_issues modest or the 300s client "
            "timeout can cut the call; retry the same call with fewer sprints "
            "instead of splitting the work across tools. Board resolution by "
            "queue/board_name follows the same rules as issues_list_sprints "
            "(board_not_found / ambiguous_board with candidate_boards and "
            "stem_candidate_boards / board_without_sprints — org sprint boards "
            "are named after the product, not the queue key; never invent a "
            "queue key out of a board name). This is NOT release and "
            "NOT per-issue history: for version_id use the release tools, for "
            "status-change evidence use issues_list_assignee_status_activity."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_metrics_sprint_history(
        ctx: Context[Any, AppContext],
        queue: Annotated[
            str | None,
            Field(
                description=(
                    "Queue key used to find the board (for example YOURQUEUE); "
                    "optional when board_id or sprint_ids is given"
                )
            ),
        ] = None,
        board_id: Annotated[
            int | None,
            Field(
                gt=0,
                description="Explicit board id; otherwise resolved by board_name or queue",
            ),
        ] = None,
        board_name: Annotated[
            str | None,
            Field(
                max_length=128,
                description=(
                    "Part of the board or product name as the user wrote it "
                    "(matched in both directions, case-insensitively); use it "
                    "instead of inventing a queue key when the user names a "
                    "board/product only."
                ),
            ),
        ] = None,
        sprint_ids: Annotated[
            list[int] | None,
            Field(
                max_length=20,
                description=(
                    "Explicit sprint ids (any depth, any order); when given, the "
                    "board is only used for names"
                ),
            ),
        ] = None,
        last_n: Annotated[
            int,
            Field(
                ge=1,
                le=20,
                description=(
                    "How many of the board's most recent started sprints to analyse "
                    "when sprint_ids is omitted"
                ),
            ),
        ] = 5,
        include_current: Annotated[
            bool,
            Field(description="Include the in-progress sprint in the window"),
        ] = True,
        max_issues: Annotated[
            int,
            Field(ge=1, le=10_000, description="Scan ceiling PER sprint"),
        ] = 1_000,
    ) -> dict[str, Any]:
        if queue is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        app = ctx.request_context.lifespan_context
        issues_api = app.issues
        auth = get_yandex_auth(ctx)
        today = date.today().isoformat()
        mode = "explicit_ids"
        board: dict[str, Any] = {"id": board_id, "name": None}
        selected: list[dict[str, Any]] = []
        boards: list[dict[str, Any]] | None = None
        if sprint_ids:
            mode = "explicit_ids"
            for sprint_id in sprint_ids:
                try:
                    selected.append(
                        _sprint_row(await issues_api.sprint_get(sprint_id, auth=auth))
                    )
                except SprintNotFound:
                    selected.append({"id": sprint_id, "status": "sprint_not_found"})
                except Exception as exc:  # noqa: BLE001 — isolate one bad sprint
                    selected.append(
                        {
                            "id": sprint_id,
                            "status": "sprint_error",
                            "error": type(exc).__name__,
                        }
                    )
        else:
            mode = "board_last_n"
            board, error, boards = await _resolve_board(
                issues_api,
                auth,
                queue=queue,
                board_name=board_name,
                board_id=board_id,
            )
            if error is not None:
                return error
            if board is None:  # pragma: no cover - helper contract
                return {"status": "board_not_found", "complete": False}
            try:
                sprints = await issues_api.board_get_sprints(
                    _as_int(board["id"]), auth=auth
                )
            except Exception as exc:  # noqa: BLE001 — boards without sprints
                sprints = []
                sprints_error = type(exc).__name__
            else:
                sprints_error = None
            if not sprints:
                candidates = _stem_candidates(boards or [], queue) if queue else []
                return {
                    "status": "board_without_sprints",
                    "complete": False,
                    "board_id": board["id"],
                    "board_name": board.get("name"),
                    "error": sprints_error,
                    "candidate_boards": candidates,
                    "candidate_basis": "board names matching the queue product stem",
                    "required_action": (
                        "This board carries no sprints; pass board_id from "
                        "candidate_boards, or pass sprint_ids explicitly. Do not "
                        "guess a board."
                    ),
                }
            rows = [_sprint_row(sprint) for sprint in sprints]
            dated = [
                row for row in rows if row.get("end_date") or row.get("start_date")
            ]
            started = [
                row for row in dated if str(row.get("start_date") or "") <= today
            ]
            pool = started or dated
            pool.sort(
                key=lambda row: (
                    str(row.get("end_date") or row.get("start_date") or ""),
                    _as_int(row.get("id")),
                ),
                reverse=True,
            )
            if not include_current:
                pool = [row for row in pool if str(row.get("end_date") or "") < today]
            selected = pool[:last_n]
            if not selected:
                return {
                    "status": "insufficient_sprints",
                    "complete": True,
                    "mode": mode,
                    "board_id": board["id"],
                    "board_name": board.get("name"),
                    "sprints_found": len(rows),
                    "required_action": (
                        "Report that the board has no started sprint in scope; do "
                        "not substitute another board or release."
                    ),
                }
        per_sprint: list[dict[str, Any]] = []
        for row in selected:
            if row.get("status") in {"sprint_not_found", "sprint_error"}:
                per_sprint.append(dict(row))
                continue
            sprint_id = _as_int(row["id"])
            entry: dict[str, Any] = {
                "id": sprint_id,
                "name": row.get("name"),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "state": row.get("state"),
            }
            try:
                metrics = await _sprint_metrics(
                    issues_api, auth, sprint_id, queue, max_issues
                )
            except Exception as exc:  # noqa: BLE001 — isolate one bad sprint
                entry["status"] = "sprint_error"
                entry["error"] = type(exc).__name__
                per_sprint.append(entry)
                continue
            entry.update(
                {
                    "status": "complete" if metrics["drain_complete"] else "sampled",
                    "issues_total": metrics["issues_total"],
                    "done": metrics["done"],
                    "remaining": metrics["remaining"],
                    "completion_share": metrics["completion_share"],
                    "plan_hours": metrics["plan_hours"],
                    "fact_hours": metrics["fact_hours"],
                    "status_type_counts": metrics["status_type_counts"],
                    "sampled": not metrics["drain_complete"],
                }
            )
            per_sprint.append(entry)
        per_sprint.sort(
            key=lambda item: (str(item.get("end_date") or ""), _as_int(item.get("id")))
        )
        with_data = [
            entry
            for entry in per_sprint
            if entry.get("status") not in _SPRINT_FAILURE_STATES
        ]
        with_issues = [entry for entry in with_data if entry.get("issues_total")]
        issues_total = sum(entry.get("issues_total", 0) for entry in with_data)
        done_total = sum(entry.get("done", 0) for entry in with_data)
        plan_total = round(sum(entry.get("plan_hours", 0.0) for entry in with_data), 2)
        fact_total = round(sum(entry.get("fact_hours", 0.0) for entry in with_data), 2)
        shares = [
            entry["completion_share"]
            for entry in with_issues
            if entry.get("completion_share") is not None
        ]
        payload: dict[str, Any] = {
            "status": "complete",
            "complete": not any(
                entry.get("status") in _SPRINT_FAILURE_STATES for entry in per_sprint
            ),
            "mode": mode,
            "board_id": board.get("id"),
            "board_name": board.get("name"),
            "filter": {
                "queue": queue,
                "last_n": last_n,
                "include_current": include_current,
                "sprint_ids": sprint_ids,
            },
            "per_sprint": per_sprint,
            "totals": {
                "sprints_analysed": len(with_data),
                "sprints_failed": len(per_sprint) - len(with_data),
                "issues_total": issues_total,
                "done_total": done_total,
                "remaining_total": issues_total - done_total,
                "plan_hours": plan_total,
                "fact_hours": fact_total,
                "average_completion_share": (
                    round(sum(shares) / len(shares), 3) if shares else None
                ),
                "plan_fact_ratio": (
                    round(fact_total / plan_total, 3) if plan_total else None
                ),
            },
            "trend": [
                {
                    "sprint_id": entry.get("id"),
                    "name": entry.get("name"),
                    "end_date": entry.get("end_date"),
                    "issues_total": entry.get("issues_total"),
                    "done": entry.get("done"),
                    "completion_share": entry.get("completion_share"),
                    "plan_hours": entry.get("plan_hours"),
                    "fact_hours": entry.get("fact_hours"),
                }
                for entry in with_data
            ],
            "history_table": _history_table(with_data),
            "coverage": {
                "complete": all(
                    entry.get("status") == "complete" for entry in with_data
                ),
                "sprints_requested": len(selected),
                "sprints_processed": len(with_data),
                "max_issues_per_sprint": max_issues,
                "sprint_filter": "structured sprint filter via the client",
                "timeout_note": (
                    "One bounded scan per sprint; if the call times out, retry the "
                    "same call with a smaller last_n."
                ),
            },
            "reporting_contract": {
                "copy_history_table_verbatim": True,
                "hours_convention": "8h/day, 5d/week; fact = logged spent time",
                "done_means": "statusType in (done, cancelled)",
                "report_sprint_errors": True,
                "no_followup_tools": True,
            },
        }
        if mode == "board_last_n":
            payload["selection_note"] = (
                f"Took the {len(per_sprint)} most recent started sprint(s) of the "
                f"board by end date (include_current={include_current})."
            )
        return _cap_rows_for_budget(payload, "per_sprint", "trend")
