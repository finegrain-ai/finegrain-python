import logging

from quart import Quart, Response, request

from chatgpt_bridge.actions.cutout import cutout
from chatgpt_bridge.actions.erase import erase
from chatgpt_bridge.actions.recolor import recolor
from chatgpt_bridge.actions.shadow import shadow
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


@app.errorhandler(RuntimeError)
async def handle_runtime_error(error: RuntimeError) -> Response:
    app.logger.error(f"{error=}")
    return json_error(str(error))


@app.errorhandler(ValueError)
async def handle_value_error(error: ValueError) -> Response:
    app.logger.error(f"{error=}")
    return json_error(str(error))


@app.errorhandler(AssertionError)
async def handle_assertion_error(error: AssertionError) -> Response:
    app.logger.error(f"{error=}")
    return json_error(str(error))


@app.errorhandler(ExceptionGroup)
async def handle_exception_group(error: ExceptionGroup) -> Response:
    errors = [str(e) for e in error.exceptions]
    app.logger.error(f"{errors=}")
    return json_error(errors)


@app.post("/cutout")
async def _cutout() -> Response:
    async with get_ctx() as ctx:
        return await cutout(ctx, request)


@app.post("/erase")
async def _erase() -> Response:
    async with get_ctx() as ctx:
        return await erase(ctx, request)


@app.post("/recolor")
async def _recolor() -> Response:
    async with get_ctx() as ctx:
        return await recolor(ctx, request)


@app.post("/shadow")
async def _shadow() -> Response:
    async with get_ctx() as ctx:
        return await shadow(ctx, request)
