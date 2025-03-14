import logging
from importlib.metadata import version
from typing import Any

from quart import Quart, Response, request

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.env import (
    APP_LOGLEVEL,
    CHATGPT_AUTH_TOKEN,
    FG_API_PASSWORD,
    FG_API_PRIORITY,
    FG_API_TIMEOUT,
    FG_API_URL,
    FG_API_USER,
    LOGLEVEL,
)
from chatgpt_bridge.skills.cutout import _cutout
from chatgpt_bridge.skills.erase import _eraser
from chatgpt_bridge.skills.recolor import _recolor
from chatgpt_bridge.skills.shadow import _shadow
from chatgpt_bridge.skills.undo import _undo
from chatgpt_bridge.utils import json_error, require_basic_auth_token

__version__ = version("chatgpt_bridge")

ctx = EditorAPIContext(
    base_url=FG_API_URL,
    user=FG_API_USER,
    password=FG_API_PASSWORD,
    priority=FG_API_PRIORITY,
    default_timeout=FG_API_TIMEOUT,
    user_agent=f"chatgpt-bridge/{__version__}",
)

app = Quart(__name__)

logging.basicConfig(level=LOGLEVEL)
app.logger.setLevel(APP_LOGLEVEL)
app.logger.info(f"{LOGLEVEL=}")
app.logger.info(f"{FG_API_URL=}")
app.logger.info(f"{FG_API_USER=}")
app.logger.info(f"{FG_API_TIMEOUT=}")
app.logger.info(f"{FG_API_PRIORITY=}")


@app.before_serving
async def login() -> None:
    await ctx.login()


@app.before_serving
async def sse_start() -> None:
    await ctx.sse_start()


@app.after_serving
async def sse_stop() -> None:
    await ctx.sse_stop()


@app.before_request
async def log_request() -> None:
    app.logger.debug(f"Incoming request: {request.method} {request.path}")


@app.errorhandler(RuntimeError)
async def handle_runtime_error(error: RuntimeError) -> Response:
    app.logger.error(f"{error=}")
    return json_error(str(error))


@app.errorhandler(ValueError)
async def handle_value_error(error: ValueError) -> Response:
    app.logger.error(f"{error=}")
    return json_error(str(error))


@app.post("/cutout")
@require_basic_auth_token(CHATGPT_AUTH_TOKEN)
async def cutout() -> Any:
    return await _cutout(ctx, request)


@app.post("/erase")
@require_basic_auth_token(CHATGPT_AUTH_TOKEN)
async def erase() -> Any:
    return await _eraser(ctx, request)


@app.post("/recolor")
@require_basic_auth_token(CHATGPT_AUTH_TOKEN)
async def recolor() -> Any:
    return await _recolor(ctx, request)


@app.post("/shadow")
@require_basic_auth_token(CHATGPT_AUTH_TOKEN)
async def shadow() -> Any:
    return await _shadow(ctx, request)


@app.post("/undo")
@require_basic_auth_token(CHATGPT_AUTH_TOKEN)
async def undo() -> Any:
    return await _undo(ctx, request)
