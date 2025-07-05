"""The report builder, and the rules on analyst decisions."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from unravel.config import DatasetPaths
from unravel.reporting import build_results, headline


def test_report_handles_missing_results(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A partial run must still produce a report, and must say what's unmeasured."""
    monkeypatch.setattr("unravel.reporting.DatasetPaths", lambda name: DatasetPaths(name, tmp_path))
    report = build_results("x")
    assert "## Not yet measured" in report
    assert "ANTHROPIC_API_KEY" in report
    assert headline("x") == ""


def test_report_reads_parity_outcome(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    paths = DatasetPaths("x", tmp_path)
    paths.reports.mkdir(parents=True)
    (paths.reports / "parity.json").write_text(
        json.dumps(
            {
                "passed": False,
                "offline_rows": 10,
                "online_messages": 9,
                "vectors_exact": 9,
                "vectors_within_tolerance": 9,
                "tolerance": 1e-9,
                "missing": 1,
                "extra": 0,
                "conflicting_redeliveries": 0,
            }
        )
    )
    monkeypatch.setattr("unravel.reporting.DatasetPaths", lambda name: DatasetPaths(name, tmp_path))
    report = build_results("x")
    assert "**FAILED**" in report
    assert "Missing: 1" in report


def test_decision_rules() -> None:
    from unravel.api.main import Decision

    assert Decision(analyst="a", action="approve").action == "approve"
    assert Decision(analyst="a", action="reject", reason="known payroll").reason
    with pytest.raises(ValidationError, match="reason"):
        Decision(analyst="a", action="reject")
    with pytest.raises(ValidationError, match="reason"):
        Decision(analyst="a", action="reject", reason="   ")
    with pytest.raises(ValidationError):
        Decision(analyst="a", action="file")  # type: ignore[arg-type]
