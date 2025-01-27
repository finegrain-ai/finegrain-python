import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, cast

import httpx
import httpx_sse
from httpx._types import QueryParamTypes, RequestFiles

logger = logging.getLogger(__name__)

Priority = Literal["low", "standard", "high"]


class SSELoopStopped(RuntimeError):
    pass


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


class EditorAPIContext:
    user: str
    password: str
    base_url: str
    priority: Priority
    verify: bool | str
    default_timeout: float

    token: str | None
    logger: logging.Logger
    max_sse_failures: int

    _client: httpx.AsyncClient | None
    _client_ctx_depth: int
    _sse_futures: Futures[dict[str, Any]]
    _sse_task: asyncio.Task[None] | None
    _sse_failures: int
    _sse_last_event_id: str
    _sse_retry_ms: int

    def __init__(
        self,
        user: str,
        password: str,
        base_url: str = "https://api.finegrain.ai/editor",
        priority: Priority = "standard",
        verify: bool | str = True,
        default_timeout: float = 60.0,
    ) -> None:
        self.user = user
        self.password = password
        self.base_url = base_url
        self.priority = priority
        self.verify = verify
        self.default_timeout = default_timeout

        self.token = None
        self.logger = logger
        self.max_sse_failures = 5

        self._client = None
        self._client_ctx_depth = 0
        self._sse_futures = Futures()
        self._sse_task = None
        self._sse_failures = 0
        self._sse_last_event_id = ""
        self._sse_retry_ms = 0

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._client:
            assert self._client_ctx_depth > 0
            self._client_ctx_depth += 1
            return self._client
        assert self._client_ctx_depth == 0
        self._client = httpx.AsyncClient(verify=self.verify)
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

    @classmethod
    def decode_json(cls, data: str) -> dict[str, Any] | None:
        try:
            r = json.loads(data)
        except json.JSONDecodeError:
            return None
        if type(r) is not dict:
            return None
        return cast(dict[str, Any], r)

    async def _sse_loop(self) -> None:
        response = await self.request("POST", "sub-auth")
        sub_token = response.json()["token"]
        url = f"{self.base_url}/sub/{sub_token}"
        headers = {"Accept": "text/event-stream"}
        if self._sse_last_event_id:
            retry_ms = self._sse_retry_ms + 1000 * 2**self._sse_failures
            self.logger.info(f"resuming SSE from event {self._sse_last_event_id} in {retry_ms} ms")
            await asyncio.sleep(retry_ms / 1000)
            headers["Last-Event-ID"] = self._sse_last_event_id
        async with (
            httpx.AsyncClient(timeout=None, verify=self.verify) as c,
            httpx_sse.aconnect_sse(c, "GET", url, headers=headers) as es,
        ):
            es.response.raise_for_status()
            self._sse_futures["_sse_loop"].set_result({"status": "ok"})
            try:
                async for sse in es.aiter_sse():
                    self._sse_last_event_id = sse.id
                    self._sse_retry_ms = sse.retry or 0
                    jdata = self.decode_json(sse.data)
                    if (jdata is None) or ("state" not in jdata):
                        # Note: when the server restarts we typically get an
                        # empty string here, then the loop exits.
                        self.logger.warning(f"unexpected SSE data: {sse.data}")
                        continue
                    self._sse_futures[jdata["state"]].set_result(jdata)
            except asyncio.CancelledError:
                pass

    async def sse_start(self) -> None:
        assert self._sse_task is None
        self._sse_last_event_id = ""
        self._sse_retry_ms = 0
        self._sse_task = asyncio.create_task(self._sse_loop())
        assert await self.sse_await("_sse_loop")
        self._sse_failures = 0

    async def sse_recover(self) -> bool:
        while True:
            if self._sse_failures > self.max_sse_failures:
                return False
            self._sse_task = asyncio.create_task(self._sse_loop())
            try:
                assert await self.sse_await("_sse_loop")
                return True
            except SSELoopStopped:
                pass

    async def sse_stop(self) -> None:
        assert self._sse_task
        self._sse_task.cancel()
        await self._sse_task
        self._sse_task = None

    async def sse_await(self, state_id: str, timeout: float | None = None) -> bool:
        assert self._sse_task
        future = self._sse_futures[state_id]
        timeout = timeout or self.default_timeout

        while True:
            sse_task = self._sse_task
            done, _ = await asyncio.wait(
                {future, sse_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                assert state_id != "_sse_loop"
                # check if SSE is not disconnected (waited too long)
                r = await self.request("GET", f"state/meta/{state_id}", raise_for_status=False)
                if r.is_success:
                    is_ok = r.json()["status"] == "ok"
                    done, _ = await asyncio.wait(
                        {future, sse_task},
                        timeout=2.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    del self._sse_futures[state_id]
                    if done == {future}:  # we just didn't wait enough
                        return is_ok
                    self.logger.warning(f"got timeout for state {state_id} and meta is OK, restarting loop")
                    aw = asyncio.gather(sse_task, return_exceptions=True)  # catch `CancelledError` properly
                    sse_task.cancel()
                    await aw
                    return is_ok
                elif r.status_code != 404:
                    raise TimeoutError(f"state {state_id} timed out after {timeout}")
                else:
                    raise RuntimeError(f"getting state {state_id} after timeout {timeout} returned {r.status_code}")
            if sse_task in done:
                self._sse_failures += 1
                if state_id != "_sse_loop" and (await self.sse_recover()):
                    self._sse_failures = 0
                    continue
                exception = sse_task.exception()
                raise SSELoopStopped(f"SSE loop stopped while waiting for state {state_id}") from exception
            break

        assert done == {future}

        jdata = future.result()
        del self._sse_futures[state_id]
        return jdata["status"] == "ok"

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
