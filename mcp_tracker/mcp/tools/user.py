"""User-related MCP tools (read-only)."""

from typing import Annotated, Any

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field
from thefuzz import process

from mcp_tracker.mcp.context import AppContext
from mcp_tracker.mcp.errors import TrackerError
from mcp_tracker.mcp.params import PageParam, PerPageParam, UserID
from mcp_tracker.mcp.utils import get_yandex_auth
from mcp_tracker.settings import Settings
from mcp_tracker.tracker.proto.types.users import User


async def _drain_user_catalog(users_api: Any, auth: Any, max_pages: int = 100) -> list[User]:
    """Load the whole org user catalog page by page (bounded)."""
    all_users: list[User] = []
    per_page, page = 100, 1
    while page <= max_pages:
        batch = await users_api.users_list(per_page=per_page, page=page, auth=auth)
        if not batch:
            break
        all_users.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return all_users


def _user_display_name(user: User) -> str:
    if user.display:
        return str(user.display)
    return " ".join(
        part for part in (user.first_name, user.last_name) if part
    ).strip()


async def resolve_assignee_core(users_api: Any, auth: Any, query: str) -> dict[str, Any]:
    """Deterministic assignee resolution cascade (MCP-free, unit-testable).

    Priority order: exact login, exact email, unique last name, unique first
    name, exact full name (both orders), then a fuzzy full-name fallback.
    Every step that matches several people is reported as ambiguous with the
    candidate list instead of guessing.
    """
    normalized = " ".join(query.split())
    lowered = normalized.casefold()
    if len(lowered) < 2:
        return {"status": "invalid", "reason": "query is too short"}

    catalog = await _drain_user_catalog(users_api, auth)

    def exact_login() -> list[User]:
        return [
            user for user in catalog
            if user.login and user.login.strip().casefold() == lowered
        ]

    def exact_email() -> list[User]:
        return [
            user for user in catalog
            if user.email and user.email.strip().casefold() == lowered
        ]

    def exact_last_name() -> list[User]:
        return [
            user for user in catalog
            if user.last_name and user.last_name.strip().casefold() == lowered
        ]

    def exact_first_name() -> list[User]:
        return [
            user for user in catalog
            if user.first_name and user.first_name.strip().casefold() == lowered
        ]

    def exact_full_name() -> list[User]:
        hits = []
        for user in catalog:
            first = (user.first_name or "").strip().casefold()
            last = (user.last_name or "").strip().casefold()
            if not first or not last:
                continue
            if lowered in (f"{first} {last}", f"{last} {first}"):
                hits.append(user)
        return hits

    def resolved(users: list[User], method: str) -> dict[str, Any]:
        return {
            "status": "resolved",
            "login": users[0].login,
            "user": users[0],
            "method": method,
        }

    def ambiguous(users: list[User], reason: str) -> dict[str, Any]:
        ordered = sorted(users, key=lambda user: user.login or "")
        return {"status": "ambiguous", "candidates": ordered, "reason": reason}

    hits = exact_login()
    if len(hits) == 1:
        return resolved(hits, "exact_login")
    hits = exact_email()
    if len(hits) == 1:
        return resolved(hits, "exact_email")
    hits = exact_last_name()
    if len(hits) == 1:
        return resolved(hits, "exact_last_name")
    if len(hits) > 1:
        return ambiguous(hits, "several users share this last name")
    hits = exact_first_name()
    if len(hits) == 1:
        return resolved(hits, "exact_first_name")
    if len(hits) > 1:
        return ambiguous(hits, "several users share this first name")
    hits = exact_full_name()
    if len(hits) == 1:
        return resolved(hits, "exact_full_name")
    if len(hits) > 1:
        return ambiguous(hits, "several users share this full name")

    names = {idx: _user_display_name(user) for idx, user in enumerate(catalog)}
    bests = process.extractBests(lowered, names, score_cutoff=80, limit=3)
    fuzzy_hits = [catalog[idx] for _name, _score, idx in bests]
    if len(fuzzy_hits) == 1:
        return resolved(fuzzy_hits, "fuzzy_full_name")
    if len(fuzzy_hits) > 1:
        return ambiguous(fuzzy_hits, "several users approximately match")
    return {"status": "not_found"}


async def users_search_core(users_api: Any, auth: Any, query: str) -> list[User]:
    """Shared implementation of users_search (MCP-free, unit-testable).

    Exact case-insensitive match on login or email scans the catalog page by
    page and returns the single hit immediately. Otherwise fuzzy-matches the
    full name across the whole catalog (thefuzz, score cutoff 80, best 3).
    """
    normalized = query.strip().lower()

    all_users: list[User] = []
    per_page, page = 100, 1
    while True:
        batch = await users_api.users_list(
            per_page=per_page, page=page, auth=auth
        )
        if not batch:
            break
        for user in batch:
            if user.login and normalized == user.login.strip().lower():
                return [user]
            if user.email and normalized == user.email.strip().lower():
                return [user]
        all_users.extend(batch)
        page += 1

    names = {
        idx: f"{u.first_name} {u.last_name}" for idx, u in enumerate(all_users)
    }
    results = process.extractBests(normalized, names, score_cutoff=80, limit=3)
    return [all_users[idx] for _name, _score, idx in results]


def register_user_tools(_settings: Settings, mcp: FastMCP[Any]) -> None:
    """Register user-related tools (all read-only)."""

    @mcp.tool(
        title="Get All Users",
        description="Get information about user accounts registered in the organization",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def users_get_all(
        ctx: Context[Any, AppContext],
        page: PageParam = 1,
        per_page: PerPageParam = 50,
    ) -> list[User]:
        return await ctx.request_context.lifespan_context.users.users_list(
            per_page=per_page,
            page=page,
            auth=get_yandex_auth(ctx),
        )

    @mcp.tool(
        title="Search Users",
        description="Search user based on login, email or real name (first or last name, or both). "
        "Returns either single user or multiple users if several match the query or an empty list "
        "if no users matched.",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def users_search(
        ctx: Context[Any, AppContext],
        login_or_email_or_name: Annotated[
            str, Field(description="User login, email or real name to search for")
        ],
    ) -> list[User]:
        return await users_search_core(
            ctx.request_context.lifespan_context.users,
            get_yandex_auth(ctx),
            login_or_email_or_name,
        )

    @mcp.tool(
        title="Get User",
        description="Get information about a specific user by login or UID",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def user_get(
        ctx: Context[Any, AppContext],
        user_id: UserID,
    ) -> User:
        user = await ctx.request_context.lifespan_context.users.user_get(
            user_id,
            auth=get_yandex_auth(ctx),
        )
        if user is None:
            raise TrackerError(f"User `{user_id}` not found.")

        return user

    @mcp.tool(
        title="Get Current User",
        description="Get information about the current authenticated user",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def user_get_current(
        ctx: Context[Any, AppContext],
    ) -> User:
        return await ctx.request_context.lifespan_context.users.user_get_current(
            auth=get_yandex_auth(ctx),
        )
