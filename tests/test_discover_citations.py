"""Tests for citation discovery helpers."""

import subprocess
from collections.abc import Callable
from typing import Any

import pytest
import tenacity

from palit import discover_citations
from palit.discover_citations import (
    EFETCH_ATTEMPTS,
    PubmedFetchError,
    fetch_papers_by_pmids,
    strip_boolean_operators,
)


def test_strips_operator_words_regardless_of_case() -> None:
    title = "Complete absence of Cockayne syndrome group B gene product gives rise to "
    title += "UV-sensitive syndrome but not Cockayne syndrome"
    assert strip_boolean_operators(title) == (
        "Complete absence of Cockayne syndrome group B gene product gives rise to "
        "UV-sensitive syndrome but Cockayne syndrome"
    )


def test_strips_uppercase_and_mixed_case_operators() -> None:
    assert strip_boolean_operators("Alpha AND beta Or gamma NoT delta") == (
        "Alpha beta gamma delta"
    )


def test_keeps_operators_embedded_in_longer_words() -> None:
    title = "NOTCH1 and ORAI1 variants in a proband with ANDersen syndrome"
    assert strip_boolean_operators(title) == (
        "NOTCH1 ORAI1 variants in a proband with ANDersen syndrome"
    )


def test_collapses_whitespace_left_behind() -> None:
    assert strip_boolean_operators("FLNC and ADD3 variants") == "FLNC ADD3 variants"


def test_leaves_titles_without_operators_untouched() -> None:
    title = "Novel STAG3 variant causes oligoasthenoteratozoospermia"
    assert strip_boolean_operators(title) == title


class _FakeEfetch:
    """Stands in for subprocess.run: plays back *outcomes*, one per efetch call."""

    def __init__(self, outcomes: list[subprocess.CompletedProcess[bytes] | Exception]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def run(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


EMPTY_RESULT = subprocess.CompletedProcess(
    args=[], returncode=0, stdout=b"<PubmedArticleSet></PubmedArticleSet>", stderr=b""
)
FAILED = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"curl: (56)")


@pytest.fixture
def efetch(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[Any]], _FakeEfetch]:
    monkeypatch.setattr(discover_citations, "EFETCH_RETRY_WAIT", tenacity.wait_none())

    def install(outcomes: list[Any]) -> _FakeEfetch:
        fake = _FakeEfetch(outcomes)
        monkeypatch.setattr(discover_citations.subprocess, "run", fake.run)
        return fake

    return install


def test_fetch_retries_a_failed_efetch(efetch: Callable[[list[Any]], _FakeEfetch]) -> None:
    fake = efetch([FAILED, subprocess.TimeoutExpired("efetch", 300), EMPTY_RESULT])

    assert fetch_papers_by_pmids([12345678], "test") == []
    assert fake.calls == 3


def test_fetch_raises_once_retries_are_spent(efetch: Callable[[list[Any]], _FakeEfetch]) -> None:
    fake = efetch([FAILED] * EFETCH_ATTEMPTS)

    with pytest.raises(PubmedFetchError, match="curl: \\(56\\)"):
        fetch_papers_by_pmids([12345678], "test")
    assert fake.calls == EFETCH_ATTEMPTS


def test_fetch_does_not_retry_a_missing_efetch(efetch: Callable[[list[Any]], _FakeEfetch]) -> None:
    fake = efetch([FileNotFoundError("efetch")])

    with pytest.raises(FileNotFoundError):
        fetch_papers_by_pmids([12345678], "test")
    assert fake.calls == 1
