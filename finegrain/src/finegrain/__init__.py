import asyncio
import json
import logging
import random
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Literal, cast

import httpx
import httpx_sse
from httpx._types import QueryParamTypes, RequestData, RequestFiles

logger = logging.getLogger(__name__)

Priority = Literal["low", "standard", "high"]

VERSION = "0.1"


class SSELoopStopped(RuntimeError):
    first_error: Exception | None
    last_error: Exception | None

    def __init__(
        self,
        message: str | None = None,
        first_error: Exception | None = None,
        last_error: Exception | None = None,
    ) -> None:
        self.first_error = first_error
        self.last_error = last_error
        super().__init__(message or self.default_message)

    @property
    def default_message(self) -> str:
        return f"SSE loop stopped (first error: {self.first_error}, last error: {self.last_error})"


class Futures[T]:
    @classmethod
    def create_future(cls) -> asyncio.Future[T]:
        return asyncio.get_running_loop().create_future()

    def __init__(self, capacity: int = 256) -> None:
        self.futures = defaultdict[str, asyncio.Future[T]](self.create_future)
        self.capacity = capacity

    def cull(self) -> None:
        while len(self.futures) >= self.capacity:
            del self.futures[next(iter(self.futures))]

    def __getitem__(self, key: str) -> asyncio.Future[T]:
        self.cull()
        return self.futures[key]

    def __delitem__(self, key: str) -> None:
        try:
            del self.futures[key]
        except KeyError:
            pass


class RetryContext:
    max_failures: int
    max_jitter: float
    max_backoff: float
    exp_base: float
    exp_factor: float

    failures: int
    first_error: Exception | None
    last_error: Exception | None

    def __init__(
        self,
        max_failures: int = 10,
        max_jitter: float = 1.0,
        max_backoff: float = 15.0,
        exp_base: float = 2.0,
        exp_factor: float = 0.1,
    ):
        self.max_failures = max_failures
        self.max_jitter = max_jitter
        self.max_backoff = max_backoff
        self.exp_base = exp_base
        self.exp_factor = exp_factor

        self.reset()

    def reset(self) -> None:
        self.failures = 0
        self.first_error = None
        self.last_error = None

    @property
    def backoff(self) -> float:
        if self.failures == 0:
            return 0
        jitter = random.uniform(0, self.max_jitter)
        return min(self.exp_factor * (self.exp_base**self.failures) + jitter, self.max_backoff)

    @property
    def remaining_attempts(self) -> int:
        return max(self.max_failures - self.failures, 0)

    def failure(self, exc: Exception | None) -> None:
        if self.failures == 0:
            self.first_error = exc
        self.last_error = exc
        self.failures += 1

    def success(self) -> None:
        self.failures = 0


class TimeoutableAsyncIterator[T](AsyncIterator[T]):
    def __init__(self, iterator: AsyncIterator[T], timeout: float) -> None:
        self.iterator = iterator
        self.timeout = timeout

    async def __anext__(self) -> T:
        return await asyncio.wait_for(self.iterator.__anext__(), timeout=self.timeout)


class ResilientEventSource:
    get_url: Callable[[], Awaitable[str]]
    get_ping_interval: Callable[[], Awaitable[float]]
    verify: bool | str
    retry_ctx: RetryContext

    logger: logging.Logger
    server_ping_grace_period: float

    _last_event_id: str
    _retry_ms: int

    active: asyncio.Future[None]

    def __init__(
        self,
        url: str | Callable[[], Awaitable[str]],
        ping_interval: float | Callable[[], Awaitable[float]] = 0.0,
        verify: bool | str = True,
        retry_ctx: RetryContext | None = None,
    ) -> None:
        self.get_url = self.async_return(url) if isinstance(url, str) else url
        if isinstance(ping_interval, int | float):
            ping_interval = self.async_return(ping_interval)
        self.get_ping_interval = ping_interval
        self.verify = verify
        self.retry_ctx = RetryContext() if retry_ctx is None else retry_ctx

        self.logger = logger
        self.server_ping_grace_period = 3.0

    def reset(self) -> None:
        self._last_event_id = ""
        self._retry_ms = 0
        self.retry_ctx.reset()
        self.active = asyncio.get_running_loop().create_future()

    @staticmethod
    def async_return[T](x: T) -> Callable[[], Awaitable[T]]:
        async def f() -> T:
            return x

        return f

    @staticmethod
    def decode_json(data: str) -> dict[str, Any] | None:
        try:
            r = json.loads(data)
        except json.JSONDecodeError:
            return None
        if type(r) is not dict:
            return None
        return cast(dict[str, Any], r)

    @property
    def headers(self) -> dict[str, str]:
        r = {"Accept": "text/event-stream"}
        if self._last_event_id:
            r["Last-Event-ID"] = self._last_event_id
        return r

    def failure(self, exc: Exception | None) -> None:
        self.active = asyncio.get_running_loop().create_future()
        self.retry_ctx.failure(exc)

    def success(self) -> None:
        self.retry_ctx.success()
        self.active.set_result(None)

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            if self.retry_ctx.remaining_attempts == 0:
                raise SSELoopStopped(
                    first_error=self.retry_ctx.first_error,
                    last_error=self.retry_ctx.last_error,
                )
            try:
                if self.retry_ctx.failures > 0:
                    self.logger.info(
                        f"SSE loop retry attempt {self.retry_ctx.failures} "
                        f"(backoff {self.retry_ctx.backoff:.3f}, retry_ms {self._retry_ms}, "
                        f"last error {self.retry_ctx.last_error})"
                    )
                    await asyncio.sleep(self.retry_ctx.backoff + self._retry_ms / 1000)
                url = await self.get_url()
                ping_interval = await self.get_ping_interval()

                async with (
                    httpx.AsyncClient(timeout=None, verify=self.verify) as c,
                    httpx_sse.aconnect_sse(c, "GET", url, headers=self.headers) as es,
                ):
                    es.response.raise_for_status()
                    self.success()
                    if ping_interval > 0:
                        timeout = ping_interval + self.server_ping_grace_period
                        it = TimeoutableAsyncIterator(es.aiter_sse(), timeout=timeout)
                    else:
                        it = es.aiter_sse()
                    async for sse in it:
                        self._last_event_id = sse.id
                        self._retry_ms = sse.retry or 0
                        if sse.event == "ping":
                            self.logger.debug("got SSE ping")
                            continue
                        if sse.event != "message":
                            self.logger.warning(f"unexpected SSE event: {sse.event} ({sse.data})")
                            continue
                        if (event := self.decode_json(sse.data)) is None:
                            self.logger.warning(f"unexpected SSE message: {sse.data}")
                            continue
                        yield event
                    raise SSELoopStopped(message="SSE loop exited")
            except (SSELoopStopped, httpx.HTTPError, TimeoutError) as exc:
                if isinstance(exc, TimeoutError) and not str(exc):
                    exc = TimeoutError("timeout")
                self.failure(exc)


class EditorAPIContext:
    user: str
    password: str
    base_url: str
    priority: Priority
    verify: bool | str
    default_timeout: float
    user_agent: str

    token: str | None
    logger: logging.Logger

    _client: httpx.AsyncClient | None
    _client_ctx_depth: int
    _sse_futures: Futures[dict[str, Any]]
    _sse_source: ResilientEventSource
    _sse_task: asyncio.Task[None] | None
    _ping_interval: float

    def __init__(
        self,
        user: str,
        password: str,
        base_url: str = "https://api.finegrain.ai/editor",
        priority: Priority = "standard",
        verify: bool | str = True,
        default_timeout: float = 60.0,
        user_agent: str | None = None,
    ) -> None:
        self.user = user
        self.password = password
        self.base_url = base_url
        self.priority = priority
        self.verify = verify
        self.default_timeout = default_timeout

        client_ua = f"finegrain-python/{VERSION}"
        if user_agent is None:
            self.user_agent = client_ua
        else:
            self.user_agent = f"{user_agent} ({client_ua})"

        self.logger = logger
        self._sse_source = ResilientEventSource(
            url=self.get_sub_url,
            ping_interval=self.get_ping_interval,
            verify=self.verify,
        )
        self.reset()

    def reset(self) -> None:
        self.token = None
        self._client = None
        self._client_ctx_depth = 0
        self._sse_futures = Futures()
        self._sse_task = None
        self._ping_interval = 0.0
        try:
            self._sse_source.reset()
        except RuntimeError:  # outside asyncio
            pass

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._client:
            assert self._client_ctx_depth > 0
            self._client_ctx_depth += 1
            return self._client
        assert self._client_ctx_depth == 0
        self._client = httpx.AsyncClient(verify=self.verify, headers={"User-Agent": self.user_agent})
        self._client_ctx_depth = 1
        return self._client

    async def __aexit__(self, *args: Any) -> None:
        if (not self._client) or self._client_ctx_depth <= 0:
            raise RuntimeError("unbalanced __aexit__")
        self._client_ctx_depth -= 1
        if self._client_ctx_depth == 0:
            await self._client.__aexit__(*args)
            self._client = None

    @property
    def auth_headers(self) -> dict[str, str]:
        assert self.token
        return {"Authorization": f"Bearer {self.token}"}

    async def login(self) -> None:
        async with self as client:
            response = await client.post(
                f"{self.base_url}/auth/login",
                json={"username": self.user, "password": self.password},
            )
        response.raise_for_status()
        self.logger.debug(f"logged in as {self.user}")
        self.token = response.json()["token"]

    async def request(
        self,
        method: Literal["GET", "POST"],
        url: str,
        files: RequestFiles | None = None,
        data: RequestData | None = None,
        params: QueryParamTypes | None = None,
        json: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        async def _q() -> httpx.Response:
            return await client.request(
                method,
                f"{self.base_url}/{url}",
                headers=dict(headers or {}) | self.auth_headers,
                files=files,
                data=data,
                params=params,
                json=json,
            )

        async with self as client:
            r = await _q()
            if r.status_code == 401:
                self.logger.debug("renewing token")
                await self.login()
                r = await _q()

        if raise_for_status:
            r.raise_for_status()
        return r

    async def get_sub_url(self) -> str:
        response = await self.request("POST", "sub-auth")
        jdata = response.json()
        sub_token = jdata["token"]
        self._ping_interval = float(jdata.get("ping_interval", 0.0))
        return f"{self.base_url}/sub/{sub_token}"

    async def get_ping_interval(self) -> float:
        return self._ping_interval

    async def _sse_loop(self) -> None:
        async for event in self._sse_source:
            if "state" not in event:
                self.logger.warning(f"unexpected SSE message: {event}")
                continue
            self.logger.debug(f"got message: {event}")
            self._sse_futures[event["state"]].set_result(event)

    async def sse_start(self) -> None:
        assert self._sse_task is None
        self._sse_source.reset()
        self._sse_task = asyncio.create_task(self._sse_loop())
        await self._sse_source.active

    async def sse_stop(self) -> None:
        assert self._sse_task
        self._sse_task.cancel()
        exc = await asyncio.gather(self._sse_task, return_exceptions=True)
        assert len(exc) == 1 and isinstance(exc[0], asyncio.CancelledError)
        self._sse_task = None

    async def sse_await(self, state_id: str, timeout: float | None = None) -> bool:
        assert self._sse_task
        future = self._sse_futures[state_id]
        timeout = timeout or self.default_timeout

        sse_task = self._sse_task
        done, _ = await asyncio.wait(
            {future, self._sse_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if sse_task in done:
            exception = sse_task.exception()
            raise SSELoopStopped(f"SSE loop stopped while waiting for state {state_id}") from exception
        if not done:
            r = await self.request("GET", f"state/meta/{state_id}", raise_for_status=False)
            if r.is_success:
                status = r.json()["status"]
                self.logger.warning(f"got timeout for state {state_id}, found metadata with status {status}")
                return status == "ok"
            elif r.status_code != 404:
                raise TimeoutError(f"state {state_id} timed out after {timeout}")
            else:
                raise RuntimeError(f"getting state {state_id} after timeout {timeout} returned {r.status_code}")

        assert done == {future}

        event = future.result()
        del self._sse_futures[state_id]
        return event["status"] == "ok"

    async def get_meta(self, state_id: str) -> dict[str, Any]:
        response = await self.request("GET", f"state/meta/{state_id}")
        return response.json()

    async def _run_one[Tin, Tout](
        self,
        co: Callable[["EditorAPIContext", Tin], Awaitable[Tout]],
        params: Tin,
    ) -> Tout:
        # This wraps the coroutine in the SSE loop.
        # This is mostly useful if you use synchronous Python,
        # otherwise you can call the functions directly.
        if not self.token:
            await self.login()
        await self.sse_start()
        try:
            r = await co(self, params)
            return r
        finally:
            await self.sse_stop()

    def run_one_sync[Tin, Tout](
        self,
        co: Callable[["EditorAPIContext", Tin], Awaitable[Tout]],
        params: Tin,
    ) -> Tout:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._run_one(co, params))

    async def call_skill(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[str, bool]:
        params = {"priority": self.priority} | (params or {})
        response = await self.request("POST", f"skills/{url}", json=params)
        state_id = response.json()["state"]
        status = await self.sse_await(state_id, timeout=timeout)
        return state_id, status

    async def ensure_skill(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        st, ok = await self.call_skill(url, params, timeout=timeout)
        if ok:
            return st
        meta = await self.get_meta(st)
        raise RuntimeError(f"skill {url} failed with {st}: {meta}")
