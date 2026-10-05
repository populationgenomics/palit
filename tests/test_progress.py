"""Tests for palit.progress: progress lines on a console that is not a terminal."""

import io

import pytest
from rich.console import Console

from palit.progress import LoggingProgress


@pytest.mark.parametrize("step", ["advance", "update"])
def test_progress_is_logged_when_not_on_a_terminal(step: str) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    with LoggingProgress(console=console) as progress:
        task = progress.add_task("Mapping associations", total=4)
        for _ in range(4):
            if step == "advance":
                progress.advance(task)
            else:
                progress.update(task, advance=1)

    logged = output.getvalue()
    assert "Mapping associations 1/4 (25%)" in logged
    assert "Mapping associations 4/4 (100%)" in logged
