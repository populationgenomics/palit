"""Rich Progress subclass that logs progress lines in non-TTY environments."""

from typing import Any

from rich.progress import Progress, ProgressColumn, TaskID


class LoggingProgress(Progress):
    """Progress bar that also emits console.log lines when not on a TTY.

    Logs whenever the completed percentage advances by more than 1% since the
    last log, so batch-job log files get useful periodic updates without flooding.
    """

    def __init__(self, *columns: ProgressColumn | str, **kwargs: Any) -> None:
        super().__init__(*columns, **kwargs)
        self._last_logged_pct: dict[TaskID, float] = {}

    def update(self, task_id: TaskID, **kwargs: Any) -> None:
        super().update(task_id, **kwargs)

        if self.console.is_terminal:
            return

        task = self._tasks[task_id]
        if task.total is None or task.total == 0:
            return

        pct = task.completed / task.total * 100
        last_pct = self._last_logged_pct.get(task_id, -1.0)

        if pct - last_pct > 1.0 or (pct >= 100.0 and last_pct < 100.0):
            self._last_logged_pct[task_id] = pct
            self.console.log(f"{task.description} {task.completed}/{task.total} ({pct:.0f}%)")
