from contextlib import asynccontextmanager

from finegrain import EditorAPIContext
from quart import Response, jsonify
from quart import current_app as app

from figma_bridge.env import (
    FG_API_PRIORITY,
    FG_API_TIMEOUT,
    FG_API_URL,
    USER_AGENT,
)


def json_error(message: str, status: int = 400) -> Response:
    response = jsonify(error=message)
    response.status_code = status
    app.logger.error(message)
    return response


@asynccontextmanager
async def get_ctx(api_key: str):
    # create a new context
    ctx = EditorAPIContext(
        api_key=api_key,
        base_url=FG_API_URL,
        priority=FG_API_PRIORITY,
        default_timeout=FG_API_TIMEOUT,
        user_agent=USER_AGENT,
    )

    # login to the API
    await ctx.login()

    # start the sse loop
    await ctx.sse_start()

    # yield the context
    yield ctx

    # stop the sse loop
    await ctx.sse_stop()

    # clear the token
    ctx.token = None
