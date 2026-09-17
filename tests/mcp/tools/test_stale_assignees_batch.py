"""Synthetic contract for the multi-assignee stale-status aggregate."""

from datetime import datetime, timezone

from mcp_tracker.mcp.tools.issue_read import (
    issues_list_stale_assignees_work_statuses_core,
)
from mcp_tracker.tracker.proto.types.issues import Issue
from mcp_tracker.tracker.proto.types.refs import StatusReference
from mcp_tracker.tracker.proto.types.users import User


def make_user(uid: int, login: str, first: str, last: str) -> User:
    return User(
        uid=uid,
        login=login,
        firstName=first,
        lastName=last,
        display=f"{last}, {first}",
    )


USERS = [
    make_user(1, "i.ivanov", "Иван", "Иванов"),
    make_user(2, "p.petrov", "Пётр", "Петров"),
]


def make_issue(key: str, assignee_id: str, assignee_display: str) -> Issue:
    issue = Issue(key=key, summary=f"Full title for {key}")
    issue.status = StatusReference(key="testing", display="Тестируется")
    issue.statusType = {"key": "inProgress", "display": "In progress"}
    issue.queue = {"key": "TEST", "display": "TEST"}
    issue.assignee = {"id": assignee_id, "display": assignee_display}
    issue.created_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    return issue


class FakeUsers:
    calls = 0

    async def users_list(self, per_page=100, page=1, auth=None):
        self.calls += 1
        return USERS if page == 1 else []


class FakeIssues:
    def __init__(self):
        self.filter_calls = 0
        self.changelog_calls = 0
        self.items = [
            make_issue("TEST-1", "1", "Иванов, Иван"),
            make_issue("TEST-2", "2", "Петров, Пётр"),
        ]

    async def issues_find_filter(
        self, filters, *, fields=None, per_page=100, page=1, auth=None
    ):
        self.filter_calls += 1
        assert filters == {"assignee": ["i.ivanov", "p.petrov"]}
        return self.items if page == 1 else []

    async def issue_get_changelog(self, issue_id, *, auth=None):
        self.changelog_calls += 1
        assignee_id = "1" if issue_id == "TEST-1" else "2"
        return [
            {
                "updatedAt": "2026-09-11T06:00:00Z",
                "fields": [
                    {
                        "field": {"id": "assignee"},
                        "from": None,
                        "to": {"id": assignee_id},
                    },
                    {
                        "field": {"id": "status"},
                        "from": {"key": "open", "display": "Новый"},
                        "to": {"key": "testing", "display": "Тестируется"},
                    },
                ],
            }
        ]


async def test_batch_loads_catalogue_and_issues_once():
    users = FakeUsers()
    issues = FakeIssues()
    result = await issues_list_stale_assignees_work_statuses_core(
        issues,
        object(),
        users_api=users,
        assignees=["Иванов", "Петров"],
        stale_business_hours=8,
        as_of="2026-09-15T18:00:00Z",
    )

    assert result["status"] == "ok"
    assert result["coverage"]["complete"] is True
    assert result["coverage"]["catalog_loaded_once"] is True
    assert users.calls == 1
    assert issues.filter_calls == 1
    assert issues.changelog_calls == 2
    assert result["counts"]["resolved_assignees"] == 2
    assert result["counts"]["stale_total"] == 2
    assert {row["resolved_login"] for row in result["issues"]} == {
        "i.ivanov",
        "p.petrov",
    }
