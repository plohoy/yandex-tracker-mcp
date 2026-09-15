"""Contracts for the one-shot assignee status activity aggregate."""

from mcp_tracker.mcp.tools.issue_read import issues_list_assignee_status_activity_core
from mcp_tracker.tracker.proto.types.issues import Issue
from mcp_tracker.tracker.proto.types.refs import StatusReference
from mcp_tracker.tracker.proto.types.users import User


def make_user() -> User:
    return User(
        uid=1,
        login="i.ivanov",
        firstName="Иван",
        lastName="Иванов",
        display="Иванов, Иван",
    )


def make_issue(key: str, status: str = "Новый", status_type: str = "new") -> Issue:
    issue = Issue(key=key, summary=f"Task {key}")
    issue.status = StatusReference(key=status, display=status)
    issue.statusType = {"key": status_type, "display": status_type}
    issue.queue = {"key": "TEST", "display": "TEST"}
    issue.assignee = {"id": "i.ivanov", "display": "Иванов Иван"}
    return issue


def status_event(at: str, source: str, target: str) -> dict:
    return {
        "updatedAt": at,
        "fields": [
            {
                "field": {"id": "status"},
                "from": {"key": source, "display": source},
                "to": {"key": target, "display": target},
            }
        ],
    }


class FakeUsers:
    async def users_list(self, per_page=100, page=1, auth=None):
        return [make_user()] if page == 1 else []


class FakeIssues:
    def __init__(self):
        self.items = [
            make_issue("TEST-1", "Закрыто", "done"),
            make_issue("TEST-2", "Тестируется"),
            make_issue("TEST-3"),
            make_issue("TEST-4", "Закрыто", "done"),
        ]
        self.changelog_calls = 0

    async def issues_find_filter(
        self, filters, *, fields=None, per_page=100, page=1, auth=None
    ):
        assert filters == {"assignee": ["i.ivanov"]}
        return self.items if page == 1 else []

    async def issue_get_status_changelog(self, key, *, auth=None):
        self.changelog_calls += 1
        return {
            "TEST-1": [
                status_event("2026-09-13T12:00:00Z", "В работе", "Закрыто")
            ],
            "TEST-2": [
                status_event("2026-09-01T12:00:00Z", "В работе", "Тестируется")
            ],
            "TEST-3": [
                status_event("2026-09-12T10:00:00Z", "Новый", "В работе"),
                status_event(
                    "2026-09-14T10:00:00Z", "В работе", "Можно тестировать"
                ),
            ],
            "TEST-4": [
                status_event("2026-08-01T12:00:00Z", "В работе", "Закрыто")
            ],
        }[key]


async def test_historical_intervals_and_recent_closure_are_distinguished():
    api = FakeIssues()
    result = await issues_list_assignee_status_activity_core(
        api,
        object(),
        users_api=FakeUsers(),
        assignee="Иванов",
        days=5,
        as_of="2026-09-15",
    )

    assert result["status"] == "ok"
    assert result["coverage"]["complete"] is True
    rows = {row["key"]: row for row in result["issues"]}
    assert set(rows) == {"TEST-1", "TEST-2", "TEST-3"}
    assert rows["TEST-1"]["matched_classes"] == ["closed", "in_progress"]
    assert rows["TEST-2"]["matched_classes"] == ["testing"]
    assert rows["TEST-3"]["matched_classes"] == ["in_progress", "testing"]
    assert api.changelog_calls == 4


async def test_unknown_status_class_fails_closed():
    result = await issues_list_assignee_status_activity_core(
        FakeIssues(),
        object(),
        users_api=FakeUsers(),
        assignee="Иванов",
        status_classes=["bogus"],
    )
    assert result["status"] == "invalid_request"
