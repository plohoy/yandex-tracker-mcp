"""Issue read-only MCP tools."""

import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_tracker.mcp.context import AppContext
from mcp_tracker.mcp.params import (
    IssueID,
    IssueIDs,
    PageParam,
    PerPageParam,
    YTQuery,
)
from mcp_tracker.mcp.tools._access import check_issue_access
from mcp_tracker.mcp.tools.user import _user_display_name, resolve_assignee_core
from mcp_tracker.mcp.utils import get_yandex_auth, set_non_needed_fields_null
from mcp_tracker.settings import Settings
from mcp_tracker.tracker.proto.types.issues import (
    ChecklistItem,
    Issue,
    IssueAttachment,
    IssueFieldsEnum,
    IssueLink,
    IssueTransition,
    Worklog,
)

_LEADING_LIST_MARKER = re.compile(r"^\s*(?:[-*•]\s*)?(?:\d+\s*[.)]?\s*)?")
_EMPTY_DESCRIPTION_VALUES = {
    "-",
    "—",
    "–",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "все",
    "все модели",
    "любой",
    "любая",
    "любые",
    "независимо от модели",
    "не указано",
    "нет",
    "разные",
    "разные модели",
    "проблема гео, нерелевантно",
}

_STATUS_ALIASES = {
    "можно тестировать": "readyForTest",
    "готово к тестированию": "readyForTest",
}

_QA_READY_STATUS_RE = re.compile(
    r"(?:можно|готов(?:о|а|ы)?|готовность|ожидает|очередь).*(?:тест|qa)|"
    r"(?:тест|qa).*(?:готов|ready|queue)|ready.*(?:test|qa)",
    re.IGNORECASE,
)
_QA_ACTIVE_STATUS_RE = re.compile(
    r"^(?:тестируется|тестирование|на тестировании|проверяется|на проверке|"
    r"qa|in qa|testing|in testing)$",
    re.IGNORECASE,
)

_TRACKER_DURATION = re.compile(
    r"^P(?:(?P<weeks>\d+(?:\.\d+)?)W)?(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def _duration_hours(value: object) -> float:
    """Convert Tracker work durations to hours (8h/day, 5d/week)."""
    if value is None:
        return 0.0
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds()) / 3600
    match = _TRACKER_DURATION.fullmatch(str(value))
    if not match:
        raise ValueError(f"Unsupported Tracker duration: {value!r}")
    parts = {name: float(number or 0) for name, number in match.groupdict().items()}
    return (
        parts["weeks"] * 5 * 8
        + parts["days"] * 8
        + parts["hours"]
        + parts["minutes"] / 60
        + parts["seconds"] / 3600
    )


# In-session tool results are truncated by Hermes above a context-scaled
# budget (39,321 chars on a 64K-token model: window*4*0.15); the sandbox
# persistence path is unavailable here, so oversized results become a broken
# 1.5K preview. Hermes counts the WRAPPED form (result-string + structured
# content + untrusted wrapper), which measures ~1.3x the raw ensure_ascii=True
# serialization used below (verified: 45,072 wrapped vs 34,684 raw on the same
# payload). Budget 28,000 raw keeps the wrapped size ~36.4K — safely under the
# threshold — so every list-returning tool caps its own rows (pre-sorted by
# relevance, tail-trimmed) and reports the cap in coverage; callers never
# mistake a partial table for a complete one.
_RESPONSE_BUDGET_CHARS: int = 28_000


def _model_dump_jsonable(value: Any) -> Any:
    """Convert pydantic models (IssueComment, Worklog, ...) to plain dicts for
    the serialized-size estimate; the MCP layer serializes them the same way."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _response_size(obj: Any) -> int:
    """Length of the response as the MCP layer will serialize it into the
    conversation (ensure_ascii=True — Cyrillic inflates ~2.3x via \\uXXXX)."""
    return len(json.dumps(obj, default=_model_dump_jsonable))


def _cap_rows_for_budget(
    response: dict[str, Any],
    *list_keys: str,
    budget: int = _RESPONSE_BUDGET_CHARS,
) -> dict[str, Any]:
    """Trim trailing rows of each ``list_keys`` field until the serialized
    response fits ``budget``. Counters stay complete; coverage gains
    rows_capped / rows_total_<key> / rows_returned_<key>."""
    if not list_keys:
        return response
    if _response_size(response) <= budget:
        return response
    coverage = response.setdefault("coverage", {})
    originally = {key: len(response.get(key) or []) for key in list_keys}
    coverage["rows_capped"] = True
    for key in list_keys:
        coverage[f"rows_total_{key}"] = originally[key]
        rows = response.get(key) or []
        if not rows:
            coverage[f"rows_returned_{key}"] = 0
            continue
        lo, hi = 0, len(rows)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            response[key] = rows[:mid]
            coverage[f"rows_returned_{key}"] = mid
            if _response_size(response) <= budget:
                lo = mid
            else:
                hi = mid - 1
        response[key] = rows[:lo]
        coverage[f"rows_returned_{key}"] = lo
    if len(list_keys) == 1 and coverage.get("total_rows"):
        # Size-capped page: expose the EFFECTIVE page size and the real page
        # count so callers can continue with page=2..pages_total using the
        # returned per_page — otherwise the remainder is unreachable
        # (e.g. per_page=200 gets capped to 31 rows yet pages_total says 1).
        returned = len(response.get(list_keys[0]) or [])
        if returned:
            coverage["per_page"] = returned
            coverage["pages_total"] = max(
                1, (coverage["total_rows"] + returned - 1) // returned
            )
    return response


def _cap_worklogs_for_budget(
    result: dict[str, list],
    budget: int = _RESPONSE_BUDGET_CHARS,
) -> tuple[dict[str, list], dict[str, int]]:
    """Trim trailing worklog entries (longest list first) until the dict fits
    ``budget``. Returns (result, {issue_key: original_total}) for the trimmed
    keys so partial data is never presented as complete."""
    if _response_size(result) <= budget:
        return result, {}
    truncated: dict[str, int] = {}
    while _response_size(result) > budget:
        key = max(result, key=lambda k: len(result[k]))
        if not result[key]:
            break
        truncated.setdefault(key, len(result[key]))
        result[key] = result[key][:-1]
    return result, truncated


def _pagination_coverage(
    rows: list[Any],
    page: int,
    per_page: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Slice ``rows`` for the requested page. With per_page=None the full set
    is returned (subject to the response-size cap). Coverage gains page,
    per_page, total_rows, pages_total so callers can walk large sets in
    separate bounded calls."""
    total = len(rows)
    if per_page is None:
        return rows, {"page": 1, "per_page": None, "total_rows": total}
    start = (page - 1) * per_page
    return rows[start : start + per_page], {
        "page": page,
        "per_page": per_page,
        "total_rows": total,
        "pages_total": max(1, (total + per_page - 1) // per_page),
    }


def _normalize_description_label(value: str) -> str:
    """Normalize a human-written form label without weakening exact matching."""
    value = _LEADING_LIST_MARKER.sub("", value)
    value = value.strip().strip("*_` ")
    value = " ".join(value.casefold().split())
    # The YOURQUEUE form used both labels over time; they represent the same
    # field, while free-form lines containing the word "модель" do not.
    if value == "производитель и модель устройства":
        return "модель устройства"
    return value


def _extract_description_field(description: str | None, field_label: str) -> list[str]:
    """Return non-empty values from exact ``label: value`` lines."""
    if not description:
        return []

    wanted = _normalize_description_label(field_label)
    values: list[str] = []
    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        separator = ":" if ":" in line else "：" if "：" in line else None
        if separator is None:
            continue
        label, value = line.split(separator, 1)
        if _normalize_description_label(label) != wanted:
            continue
        value = " ".join(value.strip().strip("*_` ").split())
        if value and value.casefold() not in _EMPTY_DESCRIPTION_VALUES:
            values.append(value)
    return values


def _summarize_description_field(
    issues: list[Issue], field_label: str, top_n: int
) -> dict[str, Any]:
    """Build a conservative, reproducible aggregate without asking an LLM."""
    grouped_keys: dict[str, list[str]] = defaultdict(list)
    spellings: dict[str, Counter[str]] = defaultdict(Counter)
    missing: list[str] = []
    multi_value: list[dict[str, Any]] = []

    for issue in issues:
        key = issue.key or "<unknown>"
        values = _extract_description_field(issue.description, field_label)
        unique_values = list(dict.fromkeys(value.casefold() for value in values))
        if not unique_values:
            missing.append(key)
            continue
        if len(unique_values) > 1:
            multi_value.append({"key": key, "values": values})

        value_by_normalized = {value.casefold(): value for value in values}
        for normalized in unique_values:
            grouped_keys[normalized].append(key)
            spellings[normalized][value_by_normalized[normalized]] += 1

    ranked = sorted(grouped_keys, key=lambda value: (-len(grouped_keys[value]), value))
    top: list[dict[str, Any]] = []
    for normalized in ranked[:top_n]:
        keys = sorted(grouped_keys[normalized])
        display = spellings[normalized].most_common(1)[0][0]
        top.append(
            {
                "value": display,
                "count": len(keys),
                "evidence": [
                    {"key": key, "url": f"https://tracker.yandex.ru/{key}"}
                    for key in keys[:10]
                ],
                "evidence_truncated": len(keys) > 10,
                "issue_keys_sha256": hashlib.sha256(
                    "\n".join(keys).encode("utf-8")
                ).hexdigest(),
            }
        )

    matched_issue_keys = {key for keys in grouped_keys.values() for key in keys}
    return {
        "matched_issues": len(matched_issue_keys),
        "matched_mentions": sum(len(keys) for keys in grouped_keys.values()),
        "missing_field_count": len(missing),
        "missing_field_examples": missing[:20],
        "multi_value_issue_count": len(multi_value),
        "multi_value_examples": multi_value[:20],
        "distinct_values": len(grouped_keys),
        "top": top,
    }


def _build_exhaustive_query(
    queue: str, date_from: date | None, date_to: date | None
) -> str:
    """Build the small supported YQL subset; models must not invent date syntax."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
        raise ValueError("queue must be a Yandex Tracker queue key such as YOURQUEUE")
    if date_from and date_to and date_from > date_to:
        raise ValueError("date_from must be earlier than or equal to date_to")

    parts = [f'"Queue": "{queue}"']
    if date_from and date_to:
        parts.append(f'"Created": "{date_from.isoformat()}".."{date_to.isoformat()}"')
    elif date_from:
        parts.append(f'"Created": >="{date_from.isoformat()}"')
    elif date_to:
        parts.append(f'"Created": <="{date_to.isoformat()}"')
    return " AND ".join(parts)


_QUEUE_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_EVIDENCE_LIMIT = 200
_SEARCH_PAGE_SIZE = 100
_SEARCH_LIMIT_DEFAULT = 10
_SEARCH_LIMIT_MAX = 20

_ASSIGNED_PAGE_SIZE = 100
_ASSIGNED_MAX_PAGES = 20
_FINAL_STATUS_TYPE_KEYS = frozenset({"done", "cancelled"})

logger = logging.getLogger(__name__)


def normalize_search_text(text: str) -> str:
    """Collapse all whitespace runs to single spaces (unicode-aware)."""
    return " ".join(text.split())


def tokenize_search_words(normalized: str) -> list[str]:
    """Extract searchable words: unicode letters/digits/underscores only.

    Punctuation and quotes act as separators, so user-supplied quotes can
    never reshape the generated YQL; each token is additionally escaped.
    """
    return re.findall(r"\w+", normalized)


# Tracker rejects quoted full-text terms that collide with YQL keywords
# (verified live: "and"/"or" -> 422; "not", "queue", "status" are fine).
_YQL_KEYWORD_TOKENS = {"and", "or"}


def drop_unsearchable_tokens(words: list[str]) -> tuple[list[str], list[str]]:
    """Split tokens into searchable terms and dropped YQL-keyword tokens."""
    kept = [word for word in words if word.casefold() not in _YQL_KEYWORD_TOKENS]
    dropped = [word for word in words if word.casefold() in _YQL_KEYWORD_TOKENS]
    return kept, dropped


def escape_yql_string(value: str) -> str:
    """Escape a value for use inside a double-quoted YQL string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_search_text_query(text: str, queue: str | None, *, mode: str = "and") -> str:
    """Build the server-side YQL for free-text search.

    Each tokenized word becomes a full-text term; Tracker matches it
    case-insensitively and applies word morphology. mode="and" requires every
    word; mode="or" recalls issues containing any word (used as a documented
    fallback when AND matches nothing). The queue filter is AND-ed separately
    so it can never be diluted by the OR group. Punctuation and quotes are
    token separators, never YQL syntax; YQL-keyword tokens (and/or) are
    dropped because Tracker rejects them as quoted full-text terms (422).
    """
    words, _dropped = drop_unsearchable_tokens(tokenize_search_words(text))
    if not words:
        raise ValueError("no searchable terms")
    terms = [f'"{escape_yql_string(word)}"' for word in words]
    if mode == "or" and len(terms) > 1:
        joined = "(" + " OR ".join(terms) + ")"
    else:
        joined = " AND ".join(terms)
    if queue:
        joined = f'{joined} AND "Queue": "{escape_yql_string(queue)}"'
    return joined


def _evidence_fragment(haystack: str, needle_words: list[str]) -> str:
    """Return a <=200-char excerpt around the first occurrence of any word."""
    haystack = " ".join(haystack.split())
    lowered = haystack.casefold()
    best = -1
    for word in needle_words:
        position = lowered.find(word.casefold())
        if position != -1 and (best == -1 or position < best):
            best = position
    if best == -1:
        return haystack[:_EVIDENCE_LIMIT]
    start = max(0, best - 60)
    fragment = haystack[start : start + _EVIDENCE_LIMIT]
    prefix = "\u2026" if start > 0 else ""
    suffix = "\u2026" if start + _EVIDENCE_LIMIT < len(haystack) else ""
    return f"{prefix}{fragment}{suffix}"[:_EVIDENCE_LIMIT]


def _search_score(
    issue: Issue,
    text: str,
    words: list[str],
) -> tuple[int, int, list[str], str | None] | None:
    """Score one issue against the search text using loaded fields only.

    Returns (tier, matched_word_count, matched_in, evidence) or None.
    Tiers: 0 exact phrase in summary; 1 all words in summary; 2 exact phrase
    in description; 3 all words across summary+description; 4 partial word
    overlap. matched_in/evidence are derived only from fields actually
    present on the issue, so description is never claimed without being loaded.
    """
    summary = issue.summary or ""
    description = issue.description or ""
    summary_cf = summary.casefold()
    description_cf = description.casefold()
    text_cf = text.casefold()
    distinct = {word.casefold() for word in words}
    in_summary = {word for word in distinct if word in summary_cf}
    in_description = {word for word in distinct if word in description_cf}
    matched_count = len(in_summary | in_description)
    if matched_count == 0:
        return None

    matched_in: list[str] = []
    if in_summary:
        matched_in.append("summary")
    if in_description:
        matched_in.append("description")

    if text_cf in summary_cf:
        return 0, matched_count, ["summary"], _evidence_fragment(summary, words)
    if in_summary == distinct:
        return 1, matched_count, ["summary"], _evidence_fragment(summary, words)
    if text_cf in description_cf:
        return 2, matched_count, ["description"], _evidence_fragment(description, words)
    if in_summary | in_description == distinct:
        source = summary if in_summary else description
        return 3, matched_count, matched_in, _evidence_fragment(source, words)
    source = summary if in_summary else description
    return 4, matched_count, matched_in, _evidence_fragment(source, words)


async def issues_search_text_core(
    issues_api: Any,
    auth: Any,
    text: str,
    queue: str | None = None,
    limit: int = _SEARCH_LIMIT_DEFAULT,
) -> dict[str, Any]:
    """Shared implementation of issues_search_text (MCP-free, unit-testable).

    Strategy: one AND full-text page; if it confirms nothing, one documented
    OR fallback page. No pagination loop, no repeated identical query, and a
    bounded set of description confirmations for candidates whose summaries
    did not match literally.
    """
    reporting_contract = {
        "use_only_returned_matches": True,
        "do_not_invent_keys": True,
        "do_not_claim_no_matches_when_incomplete": True,
    }
    normalized = normalize_search_text(text)
    queue_value = queue.strip() if isinstance(queue, str) else None
    if queue_value == "":
        queue_value = None

    def envelope(
        status: str,
        matches: list[dict[str, Any]],
        truncated: bool,
        coverage: dict[str, Any],
        matching: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "query": {"text": normalized, "queue": queue_value, "matching": matching},
            "matches": matches,
            "returned": len(matches),
            "truncated": truncated,
            "coverage": coverage,
            "reporting_contract": reporting_contract,
        }
        if error is not None:
            result["error"] = error
        return result

    raw_words = tokenize_search_words(normalized)
    words, dropped_tokens = drop_unsearchable_tokens(raw_words)
    if len(normalized) < 2 or not words or sum(len(word) for word in words) < 2:
        return envelope(
            "invalid_request",
            [],
            False,
            {"complete": False, "reason": "invalid request"},
            None,
            error="text must contain at least 2 searchable word characters after trimming",
        )
    if queue_value is not None and not _QUEUE_KEY_RE.fullmatch(queue_value):
        return envelope(
            "invalid_request",
            [],
            False,
            {"complete": False, "reason": "invalid request"},
            None,
            error="queue must be a canonical queue key such as SOMEPROJECT",
        )
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= _SEARCH_LIMIT_MAX
    ):
        return envelope(
            "invalid_request",
            [],
            False,
            {"complete": False, "reason": "invalid request"},
            None,
            error=f"limit must be an integer between 1 and {_SEARCH_LIMIT_MAX}",
        )

    def matching_text(stage_mode: str) -> str:
        if len(words) == 1:
            base = f"full-text term {words[0]!r} over summary and description"
        elif stage_mode == "and":
            base = (
                "AND of full-text terms "
                + ", ".join(repr(word) for word in words)
                + " over summary and description"
            )
        else:
            base = (
                "relaxed OR of full-text terms "
                + ", ".join(repr(word) for word in words)
                + " (exact AND matched nothing), ranked by match strength"
            )
        note = (
            "; case-insensitive, Tracker applies word morphology, reported "
            "matches are verified as literal substrings of loaded summary/description"
        )
        if dropped_tokens:
            note += (
                "; YQL keyword tokens "
                + ", ".join(repr(token) for token in dropped_tokens)
                + " are not searchable Tracker terms and were ignored"
            )
        return base + note

    base_fields = ["key", "summary", "status", "assignee", "updatedAt"]
    issued_signatures: set[tuple[str, tuple[str, ...]]] = set()
    stage = {"name": "and"}

    async def fetch(fields: list[str]) -> list[Issue]:
        query = build_search_text_query(normalized, queue_value, mode=stage["name"])
        signature = (query, tuple(fields))
        if signature in issued_signatures:
            return []
        issued_signatures.add(signature)
        print(
            f"issues_search_text: stage={stage['name']} query={query} fields={','.join(fields)}",
            file=sys.stderr,
            flush=True,
        )
        return await issues_api.issues_find(
            query=query,
            fields=fields,
            per_page=_SEARCH_PAGE_SIZE,
            page=1,
            auth=auth,
        )

    issues_by_key: dict[str, Issue] = {}
    try:
        page = await fetch(base_fields)
        if not page:
            stage["name"] = "or"
            page = await fetch(base_fields)
        pool_size = len(page)
        for issue in page:
            if issue.key and issue.key not in issues_by_key:
                issues_by_key[issue.key] = issue
        # An extra bounded fetch of the SAME page with descriptions is needed
        # only when it can change the outcome: multi-word ranking (matched_count
        # counts summary+description) or candidates not yet confirmed by their
        # summaries (possible description-only matches). matched_in/evidence are
        # never claimed without loaded data.
        needs_descriptions = len(words) > 1 or any(
            _search_score(issue, normalized, words) is None
            for issue in issues_by_key.values()
        )
        if issues_by_key and needs_descriptions:
            description_page = await fetch(["key", "description"])
            descriptions = {
                issue.key: issue.description
                for issue in description_page
                if issue.key and issue.description
            }
            for key, issue in issues_by_key.items():
                if key in descriptions:
                    issue.description = descriptions[key]
    except Exception as exc:
        return envelope(
            "upstream_error",
            [],
            False,
            {"complete": False, "reason": f"upstream error: {type(exc).__name__}"},
            matching_text(stage["name"]),
            error=type(exc).__name__,
        )

    scored: list[tuple[int, int, str, Issue, list[str], str | None]] = []
    for key, issue in issues_by_key.items():
        hit = _search_score(issue, normalized, words)
        if hit is not None:
            tier, count, matched_in, evidence = hit
            scored.append((tier, count, key, issue, matched_in, evidence))

    scored.sort(key=lambda item: (item[0], -item[1], item[2]))

    truncated = len(scored) > limit
    matches: list[dict[str, Any]] = []
    for _tier, _count, key, issue, matched_in, evidence in scored[:limit]:
        updated_at = issue.updated_at
        matches.append(
            {
                "key": key,
                "summary": issue.summary,
                "queue": key.split("-", 1)[0],
                "status": issue.status.display if issue.status else None,
                "assignee": issue.assignee.display if issue.assignee else None,
                "updated_at": (
                    updated_at.isoformat()
                    if hasattr(updated_at, "isoformat")
                    else updated_at
                ),
                "url": f"https://tracker.yandex.ru/{key}",
                "matched_in": matched_in,
                "evidence": evidence,
            }
        )

    if pool_size >= _SEARCH_PAGE_SIZE:
        coverage = {
            "complete": False,
            "reason": "server page limit reached; more matches may exist beyond the returned set",
        }
        status = "partial"
    else:
        coverage = {"complete": True, "reason": None}
        status = "ok" if matches else "no_matches"
    return envelope(status, matches, truncated, coverage, matching_text(stage["name"]))


_ASSIGNED_OPEN_DEFINITION = (
    "open = statusType key is not 'done' and not 'cancelled'; issues with a "
    "missing statusType are conservatively counted as open and reported in "
    "counts.unknown_status_type_included_as_open"
)


def _reference_key(value: Any) -> str | None:
    if isinstance(value, dict):
        key = value.get("key")
    else:
        key = getattr(value, "key", None)
    return str(key) if key else None


def _reference_display(value: Any) -> str | None:
    if isinstance(value, dict):
        display = value.get("display")
    else:
        display = getattr(value, "display", None)
    return str(display) if display else None


def _to_utc(value: Any) -> Any:
    """Normalize a datetime to UTC-aware for cutoff comparison (naive = UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_updated_before(value: str) -> Any:
    """Parse an ISO date/datetime cutoff; fail-closed on garbage input."""
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"updated_before must be an ISO date or datetime, got {value!r}"
        ) from None
    return _to_utc(parsed)


async def _open_issues_by_person_core(
    issues_api: Any,
    auth: Any,
    *,
    users_api: Any | None = None,
    person: str | None = None,
    role: str,
    updated_before: str | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict[str, Any]:
    """Shared implementation of issues_assigned_open / issues_created_open.

    MCP-free and unit-testable. Resolves the person (the current Tracker user
    unless `person` is given; it accepts a last/first name, a login, or an
    email and is resolved through the deterministic user-catalog cascade),
    fetches every issue where that login plays `role` ("assignee" or
    "createdBy") through the structured Tracker filter (no YQL is built or
    accepted), drains pages of 100 internally up to _ASSIGNED_MAX_PAGES, and
    reports the non-final ones. The envelope shape is identical for both
    roles; only the query field name, the counts-total label, the ambiguity
    status name, and the filter key differ.
    """
    if role == "assignee":
        filter_key = "assignee"
        query_field = "assignee"
        total_label = "assigned_total"
        person_noun = "assignee"
        ambiguous_status = "ambiguous_assignee"
        log_name = "issues_assigned_open"
    elif role == "creator":
        filter_key = "createdBy"
        query_field = "creator"
        total_label = "created_total"
        person_noun = "creator"
        ambiguous_status = "ambiguous_creator"
        log_name = "issues_created_open"
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"unsupported role: {role}")

    reporting_contract = {
        "use_only_returned_matches": True,
        "do_not_invent_keys": True,
        "report_counts_verbatim": True,
    }

    def envelope(
        status: str,
        matches: list[dict[str, Any]],
        coverage: dict[str, Any],
        counts: dict[str, Any] | None = None,
        error: str | None = None,
        resolved_login: str | None = None,
        resolved_by: str | None = None,
        candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "query": {
                query_field: resolved_login,
                "resolved_by": resolved_by,
                "open_definition": _ASSIGNED_OPEN_DEFINITION,
            },
            "matches": matches,
            "returned": len(matches),
            "counts": counts,
            "coverage": coverage,
            "reporting_contract": reporting_contract,
        }
        if error is not None:
            result["error"] = error
        if candidates is not None:
            result["candidates"] = candidates
        return _cap_rows_for_budget(result, "matches")

    login: str
    resolved_by: str | None
    raw_person = person.strip() if isinstance(person, str) else ""
    if raw_person:
        if len(raw_person) > 128:
            return envelope(
                "invalid_request",
                [],
                {"complete": False, "reason": "invalid request"},
                error=f"{query_field} is too long (max 128 characters)",
                resolved_login=raw_person,
            )
        if users_api is None:
            # Degraded path (no user catalog): only a syntactically valid
            # login is accepted as-is. Production always has users_api.
            if re.fullmatch(r"[A-Za-z0-9_.-]{2,}", raw_person):
                login = raw_person
                resolved_by = "login"
            else:
                return envelope(
                    "invalid_request",
                    [],
                    {"complete": False, "reason": "invalid request"},
                    error=f"{query_field} must be a Tracker login when the user catalog is unavailable",
                    resolved_login=raw_person,
                )
        else:
            try:
                resolution = await resolve_assignee_core(users_api, auth, raw_person)
            except Exception as exc:
                return envelope(
                    "upstream_error",
                    [],
                    {
                        "complete": False,
                        "reason": f"upstream error: {type(exc).__name__}",
                    },
                    error=type(exc).__name__,
                    resolved_login=raw_person,
                )
            resolution_status = resolution.get("status")
            if resolution_status == "resolved":
                login = str(resolution.get("login") or "").strip()
                resolved_by = resolution.get("method")
            elif resolution_status == "ambiguous":
                options = [
                    {
                        "login": user.login,
                        "display": user.display or _user_display_name(user),
                    }
                    for user in resolution.get("candidates") or []
                ]
                return envelope(
                    ambiguous_status,
                    [],
                    {
                        "complete": False,
                        "reason": f"{person_noun} matches several users",
                    },
                    error=f"{raw_person!r} matches several users; call again with the exact login",
                    resolved_login=raw_person,
                    candidates=options,
                )
            elif resolution_status == "not_found":
                return envelope(
                    "user_not_found",
                    [],
                    {
                        "complete": False,
                        "reason": f"{person_noun} could not be resolved",
                    },
                    error=f"no Tracker user matches {raw_person!r}",
                    resolved_login=raw_person,
                )
            else:
                return envelope(
                    "invalid_request",
                    [],
                    {"complete": False, "reason": "invalid request"},
                    error=resolution.get("reason")
                    or f"{query_field} query is too short",
                    resolved_login=raw_person,
                )
    else:
        resolved_by = "self"
        login = ""
        if users_api is not None:
            try:
                myself = await users_api.user_get_current(auth=auth)
                login = str(getattr(myself, "login", "") or "").strip()
            except Exception as exc:
                return envelope(
                    "upstream_error",
                    [],
                    {
                        "complete": False,
                        "reason": f"upstream error: {type(exc).__name__}",
                    },
                    error=type(exc).__name__,
                )
    if not login or not re.fullmatch(r"[A-Za-z0-9_.-]{2,}", login):
        return envelope(
            "invalid_request",
            [],
            {"complete": False, "reason": "invalid request"},
            error=f"cannot resolve the {person_noun} login",
            resolved_login=raw_person or None,
            resolved_by=resolved_by,
        )

    fields = [
        "key",
        "summary",
        "status",
        "statusType",
        "queue",
        "updatedAt",
        "assignee",
        "priority",
    ]
    issues_by_key: dict[str, Issue] = {}
    complete = True
    try:
        for page_number in range(1, _ASSIGNED_MAX_PAGES + 1):
            print(
                f"{log_name}: filter {filter_key}="
                f"{login} page={page_number} fields={','.join(fields)}",
                file=sys.stderr,
                flush=True,
            )
            batch = await issues_api.issues_find_filter(
                {filter_key: [login]},
                fields=fields,
                per_page=_ASSIGNED_PAGE_SIZE,
                page=page_number,
                auth=auth,
            )
            for issue in batch:
                if issue.key and issue.key not in issues_by_key:
                    issues_by_key[issue.key] = issue
            if len(batch) < _ASSIGNED_PAGE_SIZE:
                break
        else:
            complete = False
    except Exception as exc:
        return envelope(
            "upstream_error",
            [],
            {"complete": False, "reason": f"upstream error: {type(exc).__name__}"},
            error=type(exc).__name__,
            resolved_login=login,
        )

    open_issues: list[Issue] = []
    final_counts: Counter[str] = Counter()
    unknown_type = 0
    for issue in issues_by_key.values():
        st_key = _reference_key(getattr(issue, "statusType", None))
        if st_key in _FINAL_STATUS_TYPE_KEYS:
            final_counts[st_key] += 1
        else:
            if st_key is None:
                unknown_type += 1
            open_issues.append(issue)

    open_issues.sort(key=lambda issue: issue.key or "")
    counts = {
        total_label: len(issues_by_key),
        "open": len(open_issues),
        "final": dict(final_counts),
        "unknown_status_type_included_as_open": unknown_type,
    }
    if updated_before is not None:
        cutoff = _parse_updated_before(updated_before)
        open_issues = [
            issue
            for issue in open_issues
            if issue.updated_at is None or _to_utc(issue.updated_at) < cutoff
        ]
    matches: list[dict[str, Any]] = []
    for issue in open_issues:
        updated_at = issue.updated_at
        status_type = getattr(issue, "statusType", None)
        assignee_ref = getattr(issue, "assignee", None)
        priority_ref = getattr(issue, "priority", None)
        matches.append(
            {
                "key": issue.key,
                "summary": issue.summary,
                "queue": _reference_key(getattr(issue, "queue", None)),
                "status": issue.status.display if issue.status else None,
                "status_type": {
                    "key": _reference_key(status_type),
                    "display": _reference_display(status_type),
                },
                "assignee": {
                    "id": getattr(assignee_ref, "id", None),
                    "display": _reference_display(assignee_ref),
                },
                "priority": {
                    "key": _reference_key(priority_ref),
                    "display": _reference_display(priority_ref),
                },
                "updated_at": (
                    updated_at.isoformat()
                    if hasattr(updated_at, "isoformat")
                    else updated_at
                ),
                "url": f"https://tracker.yandex.ru/{issue.key}",
            }
        )

    coverage = (
        {"complete": True, "reason": None}
        if complete
        else {
            "complete": False,
            "reason": (
                f"stopped after {_ASSIGNED_MAX_PAGES} pages of "
                f"{_ASSIGNED_PAGE_SIZE}; more issues may exist"
            ),
        }
    )
    if updated_before is not None:
        coverage["updated_before"] = updated_before
        coverage["open_after_filter"] = len(open_issues)
        coverage["filtered_out"] = counts["open"] - len(open_issues)
    matches, pagination_info = _pagination_coverage(matches, page, per_page)
    coverage.update(pagination_info)
    return envelope(
        "ok" if matches else "no_open_issues",
        matches,
        coverage,
        counts=counts,
        resolved_login=login,
        resolved_by=resolved_by,
    )


async def issues_assigned_open_core(
    issues_api: Any,
    auth: Any,
    *,
    users_api: Any | None = None,
    assignee: str | None = None,
    updated_before: str | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict[str, Any]:
    """Open issues assigned to a user (see _open_issues_by_person_core)."""
    return await _open_issues_by_person_core(
        issues_api,
        auth,
        users_api=users_api,
        person=assignee,
        role="assignee",
        updated_before=updated_before,
        page=page,
        per_page=per_page,
    )


async def issues_created_open_core(
    issues_api: Any,
    auth: Any,
    *,
    users_api: Any | None = None,
    creator: str | None = None,
    updated_before: str | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict[str, Any]:
    """Open issues created by a user (see _open_issues_by_person_core)."""
    return await _open_issues_by_person_core(
        issues_api,
        auth,
        users_api=users_api,
        person=creator,
        role="creator",
        updated_before=updated_before,
        page=page,
        per_page=per_page,
    )


def register_issue_read_tools(settings: Settings, mcp: FastMCP[Any]) -> None:
    """Register issue read-only tools."""

    @mcp.tool(
        title="Get Issue",
        description="Get a Yandex Tracker issue by its id",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issue_get(
        ctx: Context[Any, AppContext],
        issue_id: IssueID,
        include_description: Annotated[
            bool,
            Field(
                description="Whether to include issue description in the issues result. "
                "It can be large, so use only when needed.",
            ),
        ] = True,
    ) -> Issue:
        check_issue_access(settings, issue_id)

        issue = await ctx.request_context.lifespan_context.issues.issue_get(
            issue_id,
            auth=get_yandex_auth(ctx),
        )

        if not include_description:
            issue.description = None

        return issue

    @mcp.tool(
        title="Get Issue Comments",
        description="Get comments of a Yandex Tracker issue by its id",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issue_get_comments(
        ctx: Context[Any, AppContext],
        issue_id: IssueID,
        page: Annotated[
            int,
            Field(ge=1, description="Page number for paginated fetches"),
        ] = 1,
        per_page: Annotated[
            int | None,
            Field(
                ge=1,
                le=500,
                description=(
                    "Comments per page. Pass it together with page to walk "
                    "long comment threads in separate calls: "
                    "coverage.total_rows and coverage.pages_total tell you "
                    "when to stop. Without it a single call returns as many "
                    "comments as fit the response budget."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        check_issue_access(settings, issue_id)

        comments = await ctx.request_context.lifespan_context.issues.issue_get_comments(
            issue_id,
            auth=get_yandex_auth(ctx),
        )
        total_comments = len(comments)
        comments, pagination_info = _pagination_coverage(comments, page, per_page)
        response = {
            "status": "complete",
            "issue_id": issue_id,
            "total_comments": total_comments,
            "comments": comments,
            "coverage": {"complete": True, **pagination_info},
        }
        return _cap_rows_for_budget(response, "comments")

    @mcp.tool(
        title="Get Issue Links",
        description="Get a Yandex Tracker issue related links to other issues by its id",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issue_get_links(
        ctx: Context[Any, AppContext],
        issue_id: IssueID,
    ) -> list[IssueLink]:
        check_issue_access(settings, issue_id)

        return await ctx.request_context.lifespan_context.issues.issues_get_links(
            issue_id,
            auth=get_yandex_auth(ctx),
        )

    @mcp.tool(
        title="Find Issues",
        description="Find Yandex Tracker issues by queue and/or created date",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_find(
        ctx: Context[Any, AppContext],
        query: YTQuery,
        include_description: Annotated[
            bool,
            Field(
                description="Whether to include issue description in the issues result. It can be large, so use only when needed.",
            ),
        ] = False,
        fields: Annotated[
            list[IssueFieldsEnum] | None,
            Field(
                description="Fields to include in the response. In order to not pollute context window - select "
                "appropriate fields beforehand. Not specifying fields will return all available."
            ),
        ] = None,
        page: PageParam = 1,
        per_page: PerPageParam = 20,
    ) -> list[Issue]:
        # Keep list queries bounded even when a model requests an oversized page.
        per_page = min(per_page, 20)
        issues = await ctx.request_context.lifespan_context.issues.issues_find(
            query=query,
            per_page=per_page,
            page=page,
            auth=get_yandex_auth(ctx),
        )

        if not include_description:
            for issue in issues:
                issue.description = None  # Clear description to save context

        if fields is None:
            fields = [
                IssueFieldsEnum.key,
                IssueFieldsEnum.summary,
                IssueFieldsEnum.status,
                IssueFieldsEnum.assignee,
                IssueFieldsEnum.priority,
                IssueFieldsEnum.updated_at,
            ]

        set_non_needed_fields_null(issues, {f.name for f in fields})

        return issues

    @mcp.tool(
        title="Search Issues by Text",
        description=(
            "Find Yandex Tracker issues by words from their title or description. "
            "Pass the user's search words as-is; the tool builds the safe server "
            "query itself, verifies matches locally, and returns a compact ranked "
            "list with short evidence excerpts. Read-only; it never accepts YQL, "
            "never paginates, and never writes. Build the answer only from the "
            "returned matches; for status=no_matches report that nothing was "
            "found, and for status=partial state the incomplete coverage "
            "explicitly instead of claiming an exhaustive result."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_search_text(
        ctx: Context[Any, AppContext],
        text: Annotated[
            str,
            Field(
                min_length=2, description="Words from the issue title or description"
            ),
        ],
        queue: Annotated[
            str | None,
            Field(description="Optional canonical queue key, for example SOMEPROJECT"),
        ] = None,
        limit: Annotated[
            int,
            Field(ge=1, le=20, description="Maximum number of matches to return"),
        ] = 10,
    ) -> dict[str, Any]:
        return await issues_search_text_core(
            issues_api=ctx.request_context.lifespan_context.issues,
            auth=get_yandex_auth(ctx),
            text=text,
            queue=queue,
            limit=limit,
        )

    @mcp.tool(
        title="List Open Issues Assigned to a User",
        description=(
            "List all OPEN Yandex Tracker issues assigned to a user. Without "
            "the assignee parameter it is the current Tracker user; pass "
            "assignee to target someone else by last/first name, login, or "
            "email — the tool resolves the person itself. When the name "
            "matches several users the tool returns status=ambiguous_assignee "
            "with the candidate logins: ask the user which one they mean, "
            "then call again with that exact login. Open means the status "
            "type is not final ('done'/'cancelled'); the tool reports "
            "open_definition so the meaning is verbatim. Read-only; it never "
            "accepts or builds visible queries and never writes. Report "
            "counts.assigned_total and counts.open verbatim together with "
            "the returned matches. Every call returns ONE PAGE of 25 rows: "
            "start with page=1, then page=2..N until you have collected "
            "coverage.total_rows rows. Never call the same page twice — "
            "re-fetching data you already have bloats the context and can "
            "exceed the model's memory limit."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_assigned_open(
        ctx: Context[Any, AppContext],
        assignee: Annotated[
            str | None,
            Field(
                description=(
                    "Assignee to list open issues for: last/first name, login, "
                    "or email. Omit it to use the current Tracker user."
                ),
                max_length=128,
            ),
        ] = None,
        updated_before: Annotated[
            str | None,
            Field(
                description=(
                    "Optional ISO date/datetime cutoff (e.g. '2026-06-20'). "
                    "When set, only open issues whose updated_at is strictly "
                    "older than the cutoff are returned. The filter runs "
                    "server-side BEFORE the response-size cap, so stale-task "
                    "queries never lose rows to capping. Issues without "
                    "updated_at are kept. counts still cover all open issues; "
                    "coverage reports open_after_filter and filtered_out."
                ),
            ),
        ] = None,
        page: Annotated[
            int,
            Field(ge=1, description="Page number for paginated fetches"),
        ] = 1,
        per_page: Annotated[
            int,
            Field(
                ge=1,
                le=200,
                description=(
                    "Rows per page (default 25). Every call returns one page: "
                    "start with page=1 and continue page=2..N until you have "
                    "collected coverage.total_rows rows (coverage.pages_total "
                    "tells you how many pages exist). When continuing, reuse "
                    "the per_page value from the previous response's coverage "
                    "(it may be smaller than requested if the page was "
                    "size-capped). Never call the same page twice and never "
                    "re-fetch data you already have."
                ),
            ),
        ] = 25,
    ) -> dict[str, Any]:
        lifespan = ctx.request_context.lifespan_context
        return await issues_assigned_open_core(
            issues_api=lifespan.issues,
            auth=get_yandex_auth(ctx),
            users_api=lifespan.users,
            assignee=assignee,
            updated_before=updated_before,
            page=page,
            per_page=per_page,
        )

    @mcp.tool(
        title="List Open Issues Created by a User",
        description=(
            "List all OPEN Yandex Tracker issues created by a user ("
            "\u00ab\u0437\u0430\u0432\u0435\u0434\u0451\u043d\u043d\u044b\u0435 \u043c\u043d\u043e\u0439\u00bb / \u00ab\u0433\u0434\u0435 \u044f \u0430\u0432\u0442\u043e\u0440\u00bb). Without "
            "the creator parameter it is the current Tracker user; pass "
            "creator to target someone else by last/first name, login, or "
            "email \u2014 the tool resolves the person itself. When the name "
            "matches several users the tool returns status=ambiguous_creator "
            "with the candidate logins: ask the user which one they mean, "
            "then call again with that exact login. Open means the status "
            "type is not final ('done'/'cancelled'); the tool reports "
            "open_definition so the meaning is verbatim. Read-only; it never "
            "accepts or builds visible queries and never writes. Report "
            "counts.created_total and counts.open verbatim together with "
            "the returned matches. Every call returns ONE PAGE of 25 rows: "
            "start with page=1, then page=2..N until you have collected "
            "coverage.total_rows rows. Never call the same page twice — "
            "re-fetching data you already have bloats the context and can "
            "exceed the model's memory limit."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_created_open(
        ctx: Context[Any, AppContext],
        creator: Annotated[
            str | None,
            Field(
                description=(
                    "Creator to list open issues for: last/first name, login, "
                    "or email. Omit it to use the current Tracker user."
                ),
                max_length=128,
            ),
        ] = None,
        updated_before: Annotated[
            str | None,
            Field(
                description=(
                    "Optional ISO date/datetime cutoff (e.g. '2026-06-20'). "
                    "When set, only open issues whose updated_at is strictly "
                    "older than the cutoff are returned. The filter runs "
                    "server-side BEFORE the response-size cap, so stale-task "
                    "queries never lose rows to capping. Issues without "
                    "updated_at are kept. counts still cover all open issues; "
                    "coverage reports open_after_filter and filtered_out."
                ),
            ),
        ] = None,
        page: Annotated[
            int,
            Field(ge=1, description="Page number for paginated fetches"),
        ] = 1,
        per_page: Annotated[
            int,
            Field(
                ge=1,
                le=200,
                description=(
                    "Rows per page (default 25). Every call returns one page: "
                    "start with page=1 and continue page=2..N until you have "
                    "collected coverage.total_rows rows (coverage.pages_total "
                    "tells you how many pages exist). When continuing, reuse "
                    "the per_page value from the previous response's coverage "
                    "(it may be smaller than requested if the page was "
                    "size-capped). Never call the same page twice and never "
                    "re-fetch data you already have."
                ),
            ),
        ] = 25,
    ) -> dict[str, Any]:
        lifespan = ctx.request_context.lifespan_context
        return await issues_created_open_core(
            issues_api=lifespan.issues,
            auth=get_yandex_auth(ctx),
            users_api=lifespan.users,
            creator=creator,
            updated_before=updated_before,
            page=page,
            per_page=per_page,
        )

    @mcp.tool(
        title="Count Issues",
        description="Get the count of Yandex Tracker issues matching a query",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_count(
        ctx: Context[Any, AppContext],
        query: YTQuery,
    ) -> int:
        return await ctx.request_context.lifespan_context.issues.issues_count(
            query,
            auth=get_yandex_auth(ctx),
        )

    @mcp.tool(
        title="Count Queue Issues by Status",
        description=(
            "Count and list all issues in a queue with an exact displayed status. "
            "Use this when the user asks about a queue and does NOT explicitly say "
            "current sprint. Never add a sprint restriction that the user did not ask for."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_count_queue_status(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        status: Annotated[
            str, Field(min_length=1, description="Exact displayed status name")
        ],
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        wanted = " ".join(status.casefold().split())
        global_statuses = (
            await ctx.request_context.lifespan_context.fields.get_statuses(auth=auth)
        )
        status_keys = sorted(
            {
                item.key
                for item in global_statuses
                if " ".join(item.name.casefold().split()) == wanted
                or item.key.casefold() == wanted
            }
        )
        alias_key = _STATUS_ALIASES.get(wanted)
        if alias_key:
            status_keys = sorted(set(status_keys) | {alias_key})
        if not status_keys:
            return {
                "status": "status_not_resolved",
                "complete": False,
                "scope": "entire_queue",
                "queue": queue,
                "issue_status": status,
                "required_action": "Resolve the displayed status through get_statuses; do not report zero.",
            }
        issues: list[Issue] = []
        pages = 0
        for status_key in status_keys:
            page = 1
            while True:
                batch = await issues_api.issues_find_filter(
                    {"queue": queue, "status": status_key},
                    fields=["key", "summary", "status"],
                    per_page=100,
                    page=page,
                    auth=auth,
                )
                pages += 1
                issues.extend(batch)
                if len(issues) > max_issues:
                    raise ValueError(f"Queue exceeds max_issues={max_issues}")
                if len(batch) < 100:
                    break
                page += 1
        matched = [
            issue
            for issue in issues
            if issue.status
            and (
                issue.status.key in status_keys
                or " ".join((issue.status.display or "").casefold().split()) == wanted
            )
        ]
        matched = list({issue.key: issue for issue in matched}.values())
        rows = [
            {
                "key": issue.key,
                "summary": issue.summary,
                "status": issue.status.display,
                "url": f"https://tracker.yandex.ru/{issue.key}",
            }
            for issue in matched
        ]
        return _cap_rows_for_budget(
            {
                "status": "complete",
                "complete": True,
                "scope": "entire_queue",
                "queue": queue,
                "issue_status": status,
                "status_keys": status_keys,
                "count": len(rows),
                "table": rows,
                "issues_scanned": len(issues),
                "pages_fetched": pages,
                "reporting_contract": {
                    "do_not_claim_current_sprint": True,
                    "do_not_add_unrequested_scope": True,
                    "zero_is_valid": True,
                    "status_resolved_from_catalogue": True,
                },
            },
            "table",
        )

    @mcp.tool(
        title="List QA Workset Across Queues",
        description=(
            "Build a complete deduplicated table of all issues semantically 'in testing' "
            "across multiple canonical queue keys. This includes qa_ready (ready/queued "
            "for QA) and qa_active (currently being tested), discovers matching displayed "
            "statuses from Tracker, and paginates internally. Use this for multi-queue "
            "summary/table requests and for stale QA worksets. Pass stale_updated_before "
            "and/or sprint_ended_before to return only stale rows, with title, updated date, "
            "and machine-derived reasons. For counts-only summaries set include_table=false "
            "to avoid returning every issue row. "
            "Never substitute issues_find or infer totals from "
            "its 20-item page. Input queue keys must already be resolved through queues_get_all."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_list_qa_workset(
        ctx: Context[Any, AppContext],
        queues: Annotated[list[str], Field(min_length=1, max_length=50)],
        max_issues_per_queue: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
        stale_updated_before: Annotated[
            date | None,
            Field(description="Optional exclusive cutoff for issue updated_at."),
        ] = None,
        stale_for_days: Annotated[
            int | None,
            Field(
                ge=1,
                le=3650,
                description="Stable age threshold in days; converted to one cutoff at tool start.",
            ),
        ] = None,
        sprint_ended_before: Annotated[
            date | None,
            Field(
                description="Optional exclusive cutoff for a resolved sprint end date."
            ),
        ] = None,
        include_table: Annotated[
            bool,
            Field(
                description="Return issue rows. Set false when only counts/coverage are requested."
            ),
        ] = True,
        page: Annotated[
            int,
            Field(ge=1, description="Page number for paginated fetches"),
        ] = 1,
        per_page: Annotated[
            int | None,
            Field(
                ge=1,
                le=500,
                description=(
                    "Rows per page. Pass it together with page to walk large "
                    "worksets in separate calls: coverage.total_rows and "
                    "coverage.pages_total tell you when to stop. Without it a "
                    "single call returns as many rows as fit the response "
                    "budget (coverage.rows_capped=true when trimmed)."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        if stale_updated_before and stale_for_days:
            raise ValueError("Use stale_updated_before or stale_for_days, not both")
        effective_updated_before = (
            date.today() - timedelta(days=stale_for_days)
            if stale_for_days
            else stale_updated_before
        )
        canonical = []
        for queue in queues:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
                raise ValueError(f"Invalid canonical queue key: {queue!r}")
            key = queue.upper()
            if key not in canonical:
                canonical.append(key)

        app = ctx.request_context.lifespan_context
        auth = get_yandex_auth(ctx)
        statuses = await app.fields.get_statuses(auth=auth)
        semantic_statuses: list[tuple[str, str, str]] = []
        for status in statuses:
            display = " ".join(status.name.split())
            probe = f"{display} {status.key}"
            if _QA_READY_STATUS_RE.search(probe):
                semantic_statuses.append((status.key, display, "qa_ready"))
            elif _QA_ACTIVE_STATUS_RE.fullmatch(
                display
            ) or _QA_ACTIVE_STATUS_RE.fullmatch(status.key):
                semantic_statuses.append((status.key, display, "qa_active"))

        rows_by_key: dict[str, dict[str, Any]] = {}
        coverage: dict[str, dict[str, Any]] = {}
        resolved_by_queue: dict[str, list[dict[str, str]]] = {}
        for queue in canonical:
            scanned = 0
            pages = 0
            used_statuses: list[dict[str, str]] = []
            for status_key, display, semantic_class in semantic_statuses:
                fetch_page = 1
                status_matched = False
                while True:
                    batch = await app.issues.issues_find_filter(
                        {"queue": queue, "status": status_key},
                        fields=[
                            "key",
                            "summary",
                            "status",
                            "assignee",
                            "sprint",
                            "updatedAt",
                        ],
                        per_page=100,
                        page=fetch_page,
                        auth=auth,
                    )
                    pages += 1
                    scanned += len(batch)
                    if scanned > max_issues_per_queue:
                        raise ValueError(
                            f"Queue {queue} exceeds max_issues_per_queue={max_issues_per_queue}"
                        )
                    for issue in batch:
                        status_matched = True
                        rows_by_key[issue.key] = {
                            "queue": queue,
                            "key": issue.key,
                            "summary": issue.summary,
                            "status": issue.status.display if issue.status else display,
                            "semantic_class": semantic_class,
                            "assignee": issue.assignee.display
                            if issue.assignee
                            else None,
                            "updated_at": (
                                issue.updated_at.isoformat()
                                if hasattr(issue.updated_at, "isoformat")
                                else str(issue.updated_at)
                                if issue.updated_at
                                else None
                            ),
                            "sprints": [
                                {"id": sprint.id, "name": sprint.display}
                                for sprint in (issue.sprint or [])
                            ],
                            "url": f"https://tracker.yandex.ru/{issue.key}",
                        }
                    if len(batch) < 100:
                        break
                    fetch_page += 1
                if status_matched:
                    used_statuses.append(
                        {
                            "key": status_key,
                            "display": display,
                            "semantic_class": semantic_class,
                        }
                    )
            resolved_by_queue[queue] = used_statuses
            coverage[queue] = {
                "complete": True,
                "issues_matched": scanned,
                "pages_fetched": pages,
            }

        wanted_sprint_ids = {
            int(sprint["id"])
            for row in rows_by_key.values()
            for sprint in row["sprints"]
            if sprint.get("id") is not None
        }
        sprint_dates: dict[int, dict[str, Any]] = {}
        sprint_board_errors: list[dict[str, Any]] = []
        if wanted_sprint_ids:
            boards = await app.issues.boards_get_all(auth=auth)
            candidate_boards: dict[int, dict[str, object]] = {}
            for queue in canonical:
                needle = queue.casefold().replace("team", "").strip("_- ")
                for board in boards:
                    board_name = (
                        str(board.get("name", "")).casefold().replace("team", "")
                    )
                    if needle and needle in board_name and board.get("id") is not None:
                        candidate_boards[int(board["id"])] = board
            for board_id, board in candidate_boards.items():
                try:
                    board_sprints = await app.issues.board_get_sprints(
                        board_id, auth=auth
                    )
                except Exception as exc:
                    sprint_board_errors.append(
                        {
                            "board_id": board_id,
                            "board_name": board.get("name"),
                            "error_type": type(exc).__name__,
                        }
                    )
                    continue
                for sprint in board_sprints:
                    sprint_id = sprint.get("id")
                    if sprint_id is None or int(sprint_id) not in wanted_sprint_ids:
                        continue
                    sprint_dates[int(sprint_id)] = {
                        "id": int(sprint_id),
                        "name": sprint.get("name"),
                        "start_date": sprint.get("startDate"),
                        "end_date": sprint.get("endDate"),
                        "board_id": board_id,
                        "board_name": board.get("name"),
                    }
        unresolved_sprint_ids: set[int] = set()
        for row in rows_by_key.values():
            resolved = []
            for sprint in row.pop("sprints"):
                sprint_id = int(sprint["id"]) if sprint.get("id") is not None else None
                details = sprint_dates.get(sprint_id) if sprint_id is not None else None
                if details:
                    resolved.append(details)
                else:
                    if sprint_id is not None:
                        unresolved_sprint_ids.add(sprint_id)
                    resolved.append(
                        {
                            "id": sprint_id,
                            "name": sprint.get("name"),
                            "start_date": None,
                            "end_date": None,
                            "dates_resolved": False,
                        }
                    )
            row["sprints"] = resolved
            row["has_sprint"] = bool(resolved)

        all_rows = sorted(
            rows_by_key.values(), key=lambda row: (row["queue"], row["key"])
        )
        stale_mode = (
            effective_updated_before is not None or sprint_ended_before is not None
        )
        rows = []
        for row in all_rows:
            reasons = []
            updated_text = row.get("updated_at")
            if effective_updated_before and updated_text:
                try:
                    if date.fromisoformat(updated_text[:10]) < effective_updated_before:
                        reasons.append(
                            f"updated_before:{effective_updated_before.isoformat()}"
                        )
                except ValueError:
                    pass
            if sprint_ended_before:
                ended = sorted(
                    {
                        sprint["end_date"][:10]
                        for sprint in row["sprints"]
                        if sprint.get("end_date")
                        and date.fromisoformat(sprint["end_date"][:10])
                        < sprint_ended_before
                    }
                )
                reasons.extend(f"sprint_ended:{value}" for value in ended)
            row["stale_reasons"] = reasons
            if not stale_mode or reasons:
                rows.append(row)
        compact_rows = []
        for row in rows:
            if not row["sprints"]:
                sprint_text = "нет спринта"
            else:
                parts = []
                for sprint in row["sprints"]:
                    name = sprint.get("name") or str(sprint.get("id") or "спринт")
                    start = sprint.get("start_date") or "?"
                    end = sprint.get("end_date") or "?"
                    parts.append(f"{name}: {start}—{end}")
                sprint_text = "; ".join(parts)
            compact = {
                "queue": row["queue"],
                "key": row["key"],
                "assignee": row["assignee"] or "не назначен",
                "sprint_dates": sprint_text,
            }
            if stale_mode:
                compact.update(
                    {
                        "title": row["summary"],
                        "updated_at": row["updated_at"],
                        "stale_reasons": row["stale_reasons"],
                    }
                )
            compact_rows.append(compact)
        counts = Counter((row["queue"], row["semantic_class"]) for row in all_rows)
        returned_counts = Counter(row["queue"] for row in rows)
        compact_rows, pagination_info = _pagination_coverage(
            compact_rows, page, per_page
        )
        coverage.update(pagination_info)
        return {
            "status": "complete",
            "complete": True,
            "scope": "entire_queues",
            "queues": canonical,
            "resolved_statuses": {
                queue: [
                    f"{item['display']} ({item['semantic_class']})" for item in items
                ]
                for queue, items in resolved_by_queue.items()
            },
            "counts": {
                queue: {
                    "qa_ready": counts[(queue, "qa_ready")],
                    "qa_active": counts[(queue, "qa_active")],
                    "total": counts[(queue, "qa_ready")] + counts[(queue, "qa_active")],
                }
                for queue in canonical
            },
            "total_unique": len(all_rows),
            "returned_total": len(rows),
            "returned_counts": {queue: returned_counts[queue] for queue in canonical},
            "filter": {
                "stale_updated_before": effective_updated_before.isoformat()
                if effective_updated_before
                else None,
                "stale_for_days": stale_for_days,
                "sprint_ended_before": sprint_ended_before.isoformat()
                if sprint_ended_before
                else None,
                "logic": "OR",
            },
            "table_columns": (
                [
                    "queue",
                    "key",
                    "title",
                    "assignee",
                    "sprint_dates",
                    "updated_at",
                    "stale_reasons",
                ]
                if stale_mode
                else ["queue", "key", "assignee", "sprint_dates"]
            ),
            "table": compact_rows if include_table else [],
            "table_omitted": not include_table,
            "coverage": coverage,
            "sprint_coverage": {
                "sprint_ids_found": len(wanted_sprint_ids),
                "sprint_ids_with_dates": len(sprint_dates),
                "unresolved_sprint_ids": sorted(unresolved_sprint_ids),
                "board_errors": sprint_board_errors,
                "complete": not unresolved_sprint_ids,
            },
            "reporting_contract": {
                "state_resolved_statuses": True,
                "state_full_coverage": True,
                "do_not_use_issues_find": True,
                "show_assignee_and_sprint_dates": True,
                "do_not_invent_missing_sprint_dates": True,
                "copy_table_values_verbatim": True,
                "never_reconstruct_truncated_rows": True,
                "returned_total_must_equal_rendered_rows": True,
                "do_not_infer_staleness_without_stale_reasons": True,
                "counts_only_must_set_include_table_false": True,
            },
        }

    @mcp.tool(
        title="Count Queue Issues in the Current Sprint by Status",
        description=(
            "Count and list issues for a queue, status, and the active sprint of its "
            "board. Use ONLY when the user explicitly says current sprint, for questions like 'how many tasks are testing in the "
            "current sprint'. Do not repeatedly call issues_find or invent YQL. "
            "An empty result is authoritative only when this tool reports complete=true."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_count_current_sprint_status(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        status: Annotated[
            str, Field(min_length=1, description="Exact displayed status name")
        ],
        board_id: Annotated[
            int | None, Field(gt=0, description="Optional explicit board id")
        ] = None,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        boards = await issues_api.boards_get_all(auth=auth)
        if board_id is None:
            needle = queue.casefold().replace("team", "").strip("_- ")
            candidates = [
                board
                for board in boards
                if needle
                and needle in str(board.get("name", "")).casefold().replace("team", "")
            ]
            primary_name = f"{needle} sprint"
            primary = [
                board
                for board in candidates
                if " ".join(str(board.get("name", "")).casefold().split())
                == primary_name
            ]
            if len(primary) == 1:
                candidates = primary
            if len(candidates) != 1:
                return {
                    "status": "ambiguous_board" if candidates else "board_not_found",
                    "complete": False,
                    "queue": queue,
                    "candidate_boards": [
                        {"id": board.get("id"), "name": board.get("name")}
                        for board in candidates
                    ],
                    "required_action": "Supply board_id; do not report a task count.",
                }
            selected = candidates[0]
            board_id = int(selected["id"])
        else:
            selected = next(
                (board for board in boards if int(board.get("id", -1)) == board_id),
                None,
            )
            if selected is None:
                raise ValueError(f"Board {board_id} not found")

        sprints = await issues_api.board_get_sprints(board_id, auth=auth)
        today = date.today().isoformat()
        active = [
            sprint
            for sprint in sprints
            if str(sprint.get("startDate", ""))
            <= today
            <= str(sprint.get("endDate", ""))
        ]
        if len(active) != 1:
            return {
                "status": "ambiguous_active_sprint"
                if active
                else "active_sprint_not_found",
                "complete": False,
                "queue": queue,
                "board": {"id": board_id, "name": selected.get("name")},
                "candidate_sprints": [
                    {"id": sprint.get("id"), "name": sprint.get("name")}
                    for sprint in active
                ],
                "required_action": "Resolve the active sprint; do not report a task count.",
            }
        sprint = active[0]
        filters: dict[str, object] = {
            "queue": queue,
            "sprint": int(sprint["id"]),
        }
        issues: list[Issue] = []
        page = 1
        while True:
            batch = await issues_api.issues_find_filter(
                filters,
                fields=["key", "summary", "status", "sprint"],
                per_page=100,
                page=page,
                auth=auth,
            )
            issues.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        wanted_status = " ".join(status.casefold().split())
        matched = [
            issue
            for issue in issues
            if issue.status
            and " ".join((issue.status.display or "").casefold().split())
            == wanted_status
        ]
        rows = [
            {
                "key": issue.key,
                "summary": issue.summary,
                "status": issue.status.display if issue.status else None,
                "url": f"https://tracker.yandex.ru/{issue.key}",
            }
            for issue in matched
        ]
        return {
            "status": "complete",
            "complete": True,
            "queue": queue,
            "board": {"id": board_id, "name": selected.get("name")},
            "sprint": {"id": sprint.get("id"), "name": sprint.get("name")},
            "issue_status": {
                "display": status,
                "matched_keys": sorted(
                    {issue.status.key for issue in matched if issue.status}
                ),
            },
            "count": len(rows),
            "table": rows,
            "pages_fetched": page,
            "issues_in_current_sprint": len(issues),
            "reporting_contract": {
                "zero_is_valid": True,
                "state_board": True,
                "do_not_retry_issues_find": True,
            },
        }

    @mcp.tool(
        title="Summarize Numeric Field for a Release",
        description=(
            "Exhaustively fetch every issue in a queue whose Fix versions contains "
            "the supplied Tracker version id, then sum a numeric issue field. Use "
            "this for release metrics such as the local field numberOfRefunds. "
            "Do not use issues_find or write YQL for this task."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_summarize_numeric_field_by_version(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        version_id: Annotated[
            int, Field(gt=0, description="Version id from queue_get_versions")
        ],
        numeric_field: Annotated[
            str,
            Field(
                min_length=1,
                description=(
                    "Full Tracker field id from queue_get_fields, for example "
                    "66af...--numberOfRefunds. Local fields require the full id."
                ),
                pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
            ),
        ],
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")

        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        filters: dict[str, object] = {"queue": queue, "fixVersions": version_id}
        page_size = 100
        page = 1
        issues: list[Issue] = []
        seen: set[str] = set()
        while True:
            batch = await issues_api.issues_find_filter(
                filters,
                fields=["key", "summary", numeric_field],
                per_page=page_size,
                page=page,
                auth=auth,
            )
            if not batch:
                break
            for issue in batch:
                if not issue.key or issue.key in seen:
                    raise RuntimeError(f"Unstable pagination at page {page}")
                seen.add(issue.key)
                issues.append(issue)
                if len(issues) > max_issues:
                    raise ValueError(f"Release exceeds max_issues={max_issues}")
            if len(batch) < page_size:
                break
            page += 1

        rows: list[dict[str, Any]] = []
        total = 0.0
        missing = 0
        for issue in issues:
            value = getattr(issue, numeric_field, None)
            if value is None:
                missing += 1
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Field {numeric_field!r} is not numeric in issue {issue.key}"
                )
            number = float(value)
            total += number
            if number:
                rows.append(
                    {
                        "key": issue.key,
                        "summary": issue.summary,
                        "value": value,
                        "url": f"https://tracker.yandex.ru/{issue.key}",
                    }
                )

        rows.sort(key=lambda row: (-float(row["value"]), row["key"]))
        return _cap_rows_for_budget(
            {
                "status": "complete",
                "filter": filters,
                "issues_in_release": len(issues),
                "issues_with_nonzero_value": len(rows),
                "issues_without_value": missing,
                "missing_value_policy": "optional empty numeric value is counted as zero",
                "sum": int(total) if total.is_integer() else total,
                "field": numeric_field,
                "evidence": rows,
                "all_issue_keys": sorted(seen),
                "all_issue_keys_sha256": hashlib.sha256(
                    "\n".join(sorted(seen)).encode("utf-8")
                ).hexdigest(),
                "coverage": {
                    "processed_issues": len(issues),
                    "complete": True,
                    "pages_fetched": page,
                },
            },
            "evidence",
            "all_issue_keys",
        )

    @mcp.tool(
        title="Summarize Effort for an Epic or Release",
        description=(
            "Exhaustively sum either planned estimation or actual spent time for "
            "all issues in an epic or release. Use this instead of issue_get_links, "
            "issue_get_worklogs, issues_find, YQL, terminal, or direct API calls. "
            "Russian 'оценка' means measure='estimation'; 'фактические трудозатраты', "
            "'затраченное время', or worklogs means measure='spent'. Never substitute "
            "spent for estimation. Returns a compact evidence table and full coverage. "
            "For 'оценка И затраченное время по задачам релиза' call TWICE: "
            "measure='estimation' and measure='spent', then take rows by the keys "
            "already known from issues_count_release_status_returns. NEVER loop "
            "issue_get/issue_get_worklogs per task for these numbers — 10+ calls of "
            "~10KB each bloats the context and hits the server memory limit."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_summarize_effort(
        ctx: Context[Any, AppContext],
        scope: Annotated[
            Literal["epic", "release"], Field(description="Aggregation scope")
        ],
        measure: Annotated[
            Literal["estimation", "spent"],
            Field(description="Planned estimate or actual time spent"),
        ],
        epic_key: Annotated[
            str | None, Field(description="Epic issue key for scope=epic")
        ] = None,
        queue: Annotated[
            str | None, Field(description="Queue key for scope=release")
        ] = None,
        version_id: Annotated[
            int | None, Field(gt=0, description="Version id for scope=release")
        ] = None,
        max_issues: Annotated[int, Field(ge=1, le=10_000)] = 2_000,
        issue_keys: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional filter: return table rows only for these issue keys "
                    "(e.g. the returned tasks already known from "
                    "issues_count_release_status_returns). Shrinks the response to "
                    "a few KB and avoids in-session truncation of large releases. "
                    "Totals still cover the whole release."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        if scope == "epic":
            if not epic_key or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_-]*-\d+", epic_key
            ):
                raise ValueError("scope=epic requires a valid epic_key")
            filters: dict[str, object] = {"epic": epic_key}
        else:
            if not queue or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
                raise ValueError("scope=release requires a valid queue key")
            if version_id is None:
                raise ValueError("scope=release requires version_id")
            filters = {"queue": queue, "fixVersions": version_id}

        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        page_size = 100
        page = 1
        issues: list[Issue] = []
        seen: set[str] = set()
        if scope == "epic":
            # Some Tracker installations model epic children as outward subtask
            # links without populating the searchable `epic` field.
            links = await issues_api.issues_get_links(epic_key, auth=auth)
            child_keys = sorted(
                {
                    link.object.key
                    for link in links
                    if link.direction == "outward"
                    and link.type is not None
                    and link.type.id == "subtask"
                    and link.object is not None
                    and link.object.key
                }
            )
            if len(child_keys) > max_issues:
                raise ValueError(f"Scope exceeds max_issues={max_issues}")
            for child_key in child_keys:
                issue = await issues_api.issue_get(child_key, auth=auth)
                seen.add(child_key)
                issues.append(issue)
        else:
            while True:
                batch = await issues_api.issues_find_filter(
                    filters,
                    fields=["key", "summary", measure],
                    per_page=page_size,
                    page=page,
                    auth=auth,
                )
                if not batch:
                    break
                for issue in batch:
                    if not issue.key or issue.key in seen:
                        raise RuntimeError(f"Unstable pagination at page {page}")
                    seen.add(issue.key)
                    issues.append(issue)
                    if len(issues) > max_issues:
                        raise ValueError(f"Scope exceeds max_issues={max_issues}")
                if len(batch) < page_size:
                    break
                page += 1

        rows: list[dict[str, Any]] = []
        total_hours = 0.0
        for issue in issues:
            raw = getattr(issue, measure, None)
            hours = _duration_hours(raw)
            total_hours += hours
            if hours:
                rows.append(
                    {
                        "key": issue.key,
                        "summary": issue.summary,
                        "hours": round(hours, 2),
                        "url": f"https://tracker.yandex.ru/{issue.key}",
                    }
                )
        rows.sort(key=lambda row: (-row["hours"], row["key"]))
        issues_with_value = len(rows)
        if issue_keys:
            key_set = set(issue_keys)
            rows = [row for row in rows if row["key"] in key_set]
        return _cap_rows_for_budget(
            {
                "status": "complete",
                "scope": scope,
                "filter": filters,
                "measure": measure,
                "measure_meaning": (
                    "planned estimate"
                    if measure == "estimation"
                    else "actual time spent"
                ),
                "total_hours": round(total_hours, 2),
                "issues_total": len(issues),
                "issues_with_value": issues_with_value,
                "issues_without_value": len(issues) - issues_with_value,
                "table": rows,
                "coverage": {
                    "processed_issues": len(issues),
                    "complete": True,
                    "pages_fetched": page,
                    "filtered_keys": bool(issue_keys),
                    "all_issue_keys_sha256": hashlib.sha256(
                        "\n".join(sorted(seen)).encode("utf-8")
                    ).hexdigest(),
                },
                "reporting_contract": {
                    "do_not_confuse_estimation_and_spent": True,
                    "state_measure_explicitly": True,
                    "report_processed_issue_count": True,
                },
            },
            "table",
        )

    @mcp.tool(
        title="Count Testing Returns in a Release",
        description=(
            "Build a complete per-issue table of status returns for every issue in a "
            "Tracker release. It uses the structured fixVersions filter and each "
            "issue's complete status changelog; never use issues_find, terminal, or "
            "direct API calls for this request. Use metric='qa_rework_cycle' for "
            "one complete Тестируется→Провал→В работе→…→Тестируется loop. "
            "Use metric='testing_rework' for "
            "testing-to-rework transitions. Use metric='repeated_work_status' when "
            "the user defines a return as visiting the same working status twice. "
            "Status rules are server-owned presets; do not invent status names. "
            "For counting queries ('посчитай возвраты', 'таблица задач с возвратами') "
            "ALWAYS pass include_evidence=false AND returned_only=true — the response "
            "shrinks ~80% (only returned tasks, no proof lists); counters and coverage "
            "stay complete. The full evidence is only needed for audits."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_count_release_status_returns(
        ctx: Context[Any, AppContext],
        queue: Annotated[str, Field(min_length=1, description="Queue key")],
        version_id: Annotated[
            int, Field(gt=0, description="Version id from queue_get_versions")
        ],
        metric: Annotated[
            Literal["qa_rework_cycle", "testing_rework", "repeated_work_status"],
            Field(
                description=(
                    "qa_rework_cycle counts each completed QA rework loop once; "
                    "testing_rework counts Тестируется→Провал and Можно тестировать→Ревью; "
                    "repeated_work_status counts each second or later visit to the same working status"
                ),
            ),
        ] = "qa_rework_cycle",
        max_issues: Annotated[int, Field(ge=1, le=500)] = 200,
        include_evidence: Annotated[
            bool,
            Field(
                description=(
                    "If false, omit the per-issue 'evidence' proof lists from the "
                    "table (compact mode for large releases). Counters, table rows, "
                    "coverage and all other fields are unaffected. Default true "
                    "keeps the full transition evidence."
                ),
            ),
        ] = True,
        returned_only: Annotated[
            bool,
            Field(
                description=(
                    "If true, return only table rows with return_count > 0 "
                    "(compact mode for counting queries). Top-level counters and "
                    "coverage still reflect the full release. Default false "
                    "returns every issue row."
                ),
            ),
        ] = False,
        page: Annotated[
            int,
            Field(ge=1, description="Page number for paginated fetches"),
        ] = 1,
        per_page: Annotated[
            int | None,
            Field(
                ge=1,
                le=500,
                description=(
                    "Rows per page. Pass it together with page to walk large "
                    "releases in separate calls: coverage.total_rows and "
                    "coverage.pages_total tell you when to stop. Without it a "
                    "single call returns as many rows as fit the response "
                    "budget (coverage.rows_capped=true when trimmed). Counters "
                    "(issues_in_release, total_returns) always cover the whole "
                    "release."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", queue):
            raise ValueError("queue must be a Yandex Tracker queue key")
        display_pairs = [
            {"from": "Тестируется", "to": "Провал"},
            {"from": "Можно тестировать", "to": "Ревью"},
        ]
        allowed_pairs = {
            (pair["from"].casefold(), pair["to"].casefold()) for pair in display_pairs
        }
        working_statuses = {
            name.casefold()
            for name in (
                "Будем делать",
                "В работе",
                "Ревью",
                "Можно тестировать",
                "Тестируется",
                "Провал",
            )
        }

        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        filters: dict[str, object] = {"queue": queue, "fixVersions": version_id}
        issues: list[Issue] = []
        seen: set[str] = set()
        fetch_page = 1
        while True:
            batch = await issues_api.issues_find_filter(
                filters,
                fields=["key", "summary"],
                per_page=100,
                page=fetch_page,
                auth=auth,
            )
            if not batch:
                break
            for issue in batch:
                if not issue.key or issue.key in seen:
                    raise RuntimeError(
                        f"Unstable issue pagination at page {fetch_page}"
                    )
                seen.add(issue.key)
                issues.append(issue)
                if len(issues) > max_issues:
                    raise ValueError(f"Release exceeds max_issues={max_issues}")
            if len(batch) < 100:
                break
            fetch_page += 1

        def _display(value: object) -> str | None:
            if not isinstance(value, dict):
                return None
            raw = value.get("display") or value.get("key") or value.get("id")
            return str(raw) if raw is not None else None

        table: list[dict[str, Any]] = []
        total_returns = 0
        for issue in issues:
            assert issue.key is not None
            changes = await issues_api.issue_get_status_changelog(issue.key, auth=auth)
            count = 0
            evidence: list[dict[str, Any]] = []
            transition_counts: Counter[str] = Counter()
            visited_work_statuses: Counter[str] = Counter()
            status_events: list[dict[str, Any]] = []
            for change in changes:
                for changed in change.get("fields", []):
                    if not isinstance(changed, dict):
                        continue
                    field = changed.get("field") or {}
                    if not isinstance(field, dict):
                        continue
                    if field.get("id") != "status" and field.get("key") != "status":
                        continue
                    source = _display(changed.get("from"))
                    target = _display(changed.get("to"))
                    if source is not None and target is not None:
                        status_events.append(
                            {
                                "from": source,
                                "to": target,
                                "updated_at": change.get("updatedAt"),
                            }
                        )
                    if metric == "qa_rework_cycle":
                        continue
                    is_return = False
                    if source is not None and target is not None:
                        if metric == "testing_rework":
                            is_return = (
                                source.casefold(),
                                target.casefold(),
                            ) in allowed_pairs
                        elif target.casefold() in working_statuses:
                            visited_work_statuses[target.casefold()] += 1
                            is_return = visited_work_statuses[target.casefold()] > 1
                    if is_return:
                        count += 1
                        transition_counts[f"{source} → {target}"] += 1
                        evidence.append(
                            {
                                "from": source,
                                "to": target,
                                "updated_at": change.get("updatedAt"),
                            }
                        )
            if metric == "qa_rework_cycle":
                # Changelog is normally chronological; sorting also makes the
                # contract deterministic for synthetic and future API variants.
                status_events.sort(key=lambda event: str(event.get("updated_at") or ""))
                phase = "waiting_for_test"
                cycle_events: list[dict[str, Any]] = []
                for event in status_events:
                    source = event["from"]
                    target = event["to"]
                    if phase == "waiting_for_test":
                        if (
                            source.casefold() == "тестируется"
                            and target.casefold() == "провал"
                        ):
                            phase = "failed"
                            cycle_events = [event]
                        elif target.casefold() == "тестируется":
                            phase = "testing"
                            cycle_events = [event]
                    elif phase == "testing":
                        if (
                            source.casefold() == "тестируется"
                            and target.casefold() == "провал"
                        ):
                            phase = "failed"
                            cycle_events.append(event)
                        elif target.casefold() == "тестируется":
                            cycle_events = [event]
                    elif phase == "failed":
                        cycle_events.append(event)
                        if target.casefold() == "в работе":
                            phase = "rework"
                    elif phase == "rework":
                        cycle_events.append(event)
                        if target.casefold() == "тестируется":
                            count += 1
                            name = "Тестируется → Провал → В работе → Тестируется"
                            transition_counts[name] += 1
                            evidence.append(
                                {
                                    "cycle": name,
                                    "started_at": cycle_events[0].get("updated_at"),
                                    "failed_at": next(
                                        (
                                            item.get("updated_at")
                                            for item in cycle_events
                                            if item["to"].casefold() == "провал"
                                        ),
                                        None,
                                    ),
                                    "returned_to_work_at": next(
                                        (
                                            item.get("updated_at")
                                            for item in cycle_events
                                            if item["to"].casefold() == "в работе"
                                        ),
                                        None,
                                    ),
                                    "retested_at": event.get("updated_at"),
                                }
                            )
                            phase = "testing"
                            cycle_events = [event]
            total_returns += count
            table.append(
                {
                    "key": issue.key,
                    "summary": issue.summary,
                    "return_count": count,
                    "transition": "; ".join(
                        f"{name} ({amount})" if amount > 1 else name
                        for name, amount in sorted(transition_counts.items())
                    )
                    or "—",
                    "transition_counts": dict(sorted(transition_counts.items())),
                    "url": f"https://tracker.yandex.ru/{issue.key}",
                    "evidence": evidence,
                }
            )

        table.sort(key=lambda row: (-row["return_count"], row["key"]))
        if not include_evidence:
            for row in table:
                row.pop("evidence", None)
        if returned_only:
            table = [row for row in table if row["return_count"] > 0]
        table, pagination_info = _pagination_coverage(table, page, per_page)
        return _cap_rows_for_budget(
            {
                "status": "complete",
                "filter": filters,
                "definition": {
                    "metric": metric,
                    "return_transitions": display_pairs
                    if metric == "testing_rework"
                    else None,
                    "working_statuses": sorted(working_statuses)
                    if metric == "repeated_work_status"
                    else None,
                    "rule": (
                        "count one completed Тестируется→Провал→В работе→…→Тестируется loop once"
                        if metric == "qa_rework_cycle"
                        else (
                            "count only the listed exact from/to pairs"
                            if metric == "testing_rework"
                            else "for each working status, count every visit after its first visit"
                        )
                    ),
                },
                "issues_in_release": len(issues),
                "issues_with_returns": sum(row["return_count"] > 0 for row in table),
                "total_returns": total_returns,
                "table": table,
                "coverage": {
                    "processed_issues": len(issues),
                    "complete": True,
                    "returned_only": returned_only,
                    **pagination_info,
                    "all_issue_keys_sha256": hashlib.sha256(
                        "\n".join(sorted(seen)).encode("utf-8")
                    ).hexdigest(),
                },
            },
            "table",
        )

    @mcp.tool(
        title="Analyze Issue Description Field Exhaustively",
        description=(
            "Exhaustively analyze an exact 'label: value' field in descriptions of ALL "
            "issues in a queue and optional inclusive date range. Use this instead of issues_find for "
            "requests containing 'all', statistics, counts, frequency tables, or top-N. "
            "The server counts first, paginates internally, rejects duplicates or partial "
            "scans, and returns only aggregates plus evidence links. In the final answer, "
            "report coverage.found_issues and coverage.processed_issues verbatim as "
            "'found N, processed N'; do not substitute matched fields or distinct values."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issues_analyze_description_field(
        ctx: Context[Any, AppContext],
        queue: Annotated[
            str,
            Field(
                description="Yandex Tracker queue key, for example YOURQUEUE",
                min_length=1,
            ),
        ],
        field_label: Annotated[
            str,
            Field(
                description=(
                    "Exact human-readable label from an issue description, with or "
                    "without its list number; for example '4. Модель устройства'."
                ),
                min_length=1,
            ),
        ],
        date_from: Annotated[
            date | None,
            Field(
                description=(
                    "Inclusive start date in YYYY-MM-DD. For 'last year', calculate the "
                    "calendar date exactly one year before today; do not use YQL functions."
                )
            ),
        ] = None,
        date_to: Annotated[
            date | None,
            Field(
                description="Inclusive end date in YYYY-MM-DD; normally today's date"
            ),
        ] = None,
        top_n: Annotated[
            int, Field(description="Number of most frequent values", ge=1, le=50)
        ] = 15,
        max_issues: Annotated[
            int,
            Field(
                description="Safety ceiling; the tool stops before fetching if count exceeds it",
                ge=1,
                le=10_000,
            ),
        ] = 2_000,
    ) -> dict[str, Any]:
        query = _build_exhaustive_query(queue, date_from, date_to)
        issues_api = ctx.request_context.lifespan_context.issues
        auth = get_yandex_auth(ctx)
        expected = await issues_api.issues_count(query, auth=auth)
        if expected > max_issues:
            raise ValueError(
                f"Query matches {expected} issues, exceeding max_issues={max_issues}. "
                "Narrow the query or explicitly raise the safety ceiling."
            )

        page_size = 100
        page = 1
        issues: list[Issue] = []
        seen_keys: set[str] = set()
        while len(issues) < expected:
            batch = await issues_api.issues_find(
                query=query,
                per_page=page_size,
                page=page,
                auth=auth,
            )
            if not batch:
                raise RuntimeError(
                    f"Incomplete scan: expected {expected}, received {len(issues)} before page {page}."
                )
            for issue in batch:
                if not issue.key:
                    raise RuntimeError(
                        f"Incomplete scan: issue without key on page {page}."
                    )
                if issue.key in seen_keys:
                    raise RuntimeError(
                        f"Unstable pagination: duplicate issue {issue.key} on page {page}."
                    )
                seen_keys.add(issue.key)
                issues.append(issue)
            page += 1

        after = await issues_api.issues_count(query, auth=auth)
        if after != expected or len(issues) != expected:
            raise RuntimeError(
                "Query changed or pagination was incomplete: "
                f"count_before={expected}, fetched={len(issues)}, count_after={after}."
            )

        result = _summarize_description_field(issues, field_label, top_n)
        keys = sorted(seen_keys)
        result.update(
            {
                "status": "complete",
                "coverage": {
                    "found_issues": expected,
                    "processed_issues": len(issues),
                    "complete": True,
                    "percent": 100.0,
                    "required_report_text": (
                        f"Found {expected} issues; processed {len(issues)} of {expected} issues."
                    ),
                },
                "reporting_contract": {
                    "ranking_basis": "exact normalized field values",
                    "copy_values_verbatim": True,
                    "do_not_merge_brands_or_models": True,
                    "do_not_recalculate_counts": True,
                    "required_caveat": (
                        "The ranking uses exact field values; brand/model variants were not merged."
                    ),
                },
                "query": query,
                "field_label": field_label,
                "count_before": expected,
                "fetched": len(issues),
                "count_after": after,
                "pages_fetched": page - 1,
                "page_size": page_size,
                "all_issue_keys_sha256": hashlib.sha256(
                    "\n".join(keys).encode("utf-8")
                ).hexdigest(),
            }
        )
        return result

    @mcp.tool(
        title="Get Issue Worklogs",
        description="Get worklogs of a Yandex Tracker issue by its id",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issue_get_worklogs(
        ctx: Context[Any, AppContext],
        issue_ids: IssueIDs,
    ) -> dict[str, Any]:
        for issue_id in issue_ids:
            check_issue_access(settings, issue_id)

        result: dict[str, list[Worklog]] = {}
        for issue_id in issue_ids:
            worklogs = (
                await ctx.request_context.lifespan_context.issues.issue_get_worklogs(
                    issue_id,
                    auth=get_yandex_auth(ctx),
                )
            )
            result[issue_id] = worklogs or []

        result, truncated_entries = _cap_worklogs_for_budget(result)
        if truncated_entries:
            result["_truncated_entries"] = truncated_entries
        return result

    @mcp.tool(
        title="Get Issue Attachments",
        description="Get attachments of a Yandex Tracker issue by its id",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issue_get_attachments(
        ctx: Context[Any, AppContext],
        issue_id: IssueID,
    ) -> list[IssueAttachment]:
        check_issue_access(settings, issue_id)

        return await ctx.request_context.lifespan_context.issues.issue_get_attachments(
            issue_id,
            auth=get_yandex_auth(ctx),
        )

    @mcp.tool(
        title="Get Issue Checklist",
        description="Get checklist items of a Yandex Tracker issue by its id",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issue_get_checklist(
        ctx: Context[Any, AppContext],
        issue_id: IssueID,
    ) -> list[ChecklistItem]:
        check_issue_access(settings, issue_id)

        return await ctx.request_context.lifespan_context.issues.issue_get_checklist(
            issue_id,
            auth=get_yandex_auth(ctx),
        )

    @mcp.tool(
        title="Get Issue Transitions",
        description="Get possible status transitions for a Yandex Tracker issue. "
        "Returns list of available transitions that can be performed on the issue.",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def issue_get_transitions(
        ctx: Context[Any, AppContext],
        issue_id: IssueID,
    ) -> list[IssueTransition]:
        check_issue_access(settings, issue_id)

        return await ctx.request_context.lifespan_context.issues.issue_get_transitions(
            issue_id,
            auth=get_yandex_auth(ctx),
        )
