import logging

from quart import Quart, Response, request

from chatgpt_bridge.actions.cutout import _cutout
from chatgpt_bridge.actions.erase import _eraser
from chatgpt_bridge.actions.recolor import _recolor
from chatgpt_bridge.actions.shadow import _shadow
from chatgpt_bridge.actions.undo import _undo
from chatgpt_bridge.env import APP_LOGLEVEL, FG_API_PRIORITY, FG_API_TIMEOUT, FG_API_URL, LOGLEVEL, USER_AGENT
from chatgpt_bridge.utils import get_ctx, json_error

app = Quart(__name__)

logging.basicConfig(level=LOGLEVEL)
app.logger.setLevel(APP_LOGLEVEL)
app.logger.info(f"{LOGLEVEL=}")
app.logger.info(f"{FG_API_URL=}")
app.logger.info(f"{FG_API_TIMEOUT=}")
app.logger.info(f"{FG_API_PRIORITY=}")
app.logger.info(f"{USER_AGENT=}")


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
async def cutout() -> Response:
    async with get_ctx() as ctx:
        return await _cutout(ctx, request)


@app.post("/erase")
async def erase() -> Response:
    async with get_ctx() as ctx:
        return await _eraser(ctx, request)


@app.post("/recolor")
async def recolor() -> Response:
    async with get_ctx() as ctx:
        return await _recolor(ctx, request)


@app.post("/shadow")
async def shadow() -> Response:
    async with get_ctx() as ctx:
        return await _shadow(ctx, request)


@app.post("/undo")
async def undo() -> Response:
    async with get_ctx() as ctx:
        return await _undo(ctx, request)
