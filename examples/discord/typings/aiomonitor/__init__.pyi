import asyncio
from types import TracebackType
from typing import Any, Self

class Monitor:
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: Any,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

def start_monitor(loop: asyncio.AbstractEventLoop, **kwargs: Any) -> Monitor: ...
