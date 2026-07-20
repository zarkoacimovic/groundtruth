from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FutureIntegration:
    name: str
    status: str
    notes: str


def planned_integrations() -> List[FutureIntegration]:
    return [
        FutureIntegration(
            name="Slack",
            status="planned",
            notes="Future intake source for channels, DMs, and triage notifications.",
        ),
        FutureIntegration(
            name="ServiceNow",
            status="planned",
            notes="Future source for incidents, service requests, and enterprise workflows.",
        ),
        FutureIntegration(
            name="Jira",
            status="planned",
            notes="Future sync target for stories, bugs, tasks, and requirements artifacts.",
        ),
    ]
