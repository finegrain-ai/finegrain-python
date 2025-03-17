import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from environs import Env

from finegrain import EditorAPIContext

env = Env()
env.read_env()

PARENT_PATH = Path(__file__).parent


# Make sure all tests run in the same event loop.
# This requires pytest-asyncio <= 0.21, later versions are broken.
@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def fgctx() -> AsyncGenerator[EditorAPIContext, None]:
    with env.prefixed("FG_API_"):
        credentials = env.str("CREDENTIALS", None)
        url = env.str("URL", "https://api.finegrain.ai/editor")
    assert credentials and url, "set FG_API_CREDENTIALS"
    ctx = EditorAPIContext(
        credentials=credentials,
        base_url=url,
        user_agent="finegrain-python-tests",
    )

    await ctx.login()
    await ctx.sse_start()
    yield ctx
    await ctx.sse_stop()


@pytest.fixture(scope="function")
def output_dir() -> str | None:
    return env.str("FG_TESTS_OUTPUT_DIR", None)


@pytest.fixture(scope="function")
def coffee_plant_bytes() -> bytes:
    with open(PARENT_PATH / "fixtures" / "coffee-plant.jpg", "rb") as f:
        return f.read()


@pytest.fixture(scope="function")
def sofa_cushion_bytes() -> bytes:
    with open(PARENT_PATH / "fixtures" / "sofa-cushion.jpg", "rb") as f:
        return f.read()
