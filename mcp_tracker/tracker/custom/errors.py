class YandexTrackerError(Exception):
    pass


class IssueNotFound(YandexTrackerError):
    def __init__(self, issue_id: str):
        super().__init__(f"Issue with ID '{issue_id}' not found.")
        self.issue_id = issue_id


class SprintNotFound(YandexTrackerError):
    def __init__(self, sprint_id: int):
        super().__init__(f"Sprint with ID '{sprint_id}' not found.")
        self.sprint_id = sprint_id
