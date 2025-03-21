import logging

from quart import Quart, Response, request
from quart_cors import cors

from figma_bridge.env import APP_LOGLEVEL, FG_API_PRIORITY, FG_API_TIMEOUT, FG_API_URL, LOGLEVEL, USER_AGENT
from figma_bridge.erase import erase
from figma_bridge.utils import json_error

app = Quart(__name__)

cors(
    app_or_blueprint=app,
    allow_origin="*",
    allow_methods=["POST"],
)

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


@app.errorhandler(AssertionError)
async def handle_assertion_error(error: AssertionError) -> Response:
    app.logger.error(f"{error=}")
    return json_error(str(error))


@app.post("/erase")
async def _erase() -> Response:
    return await erase(request)
