import logging
from functools import wraps
from pathlib import Path

import httpx

from chatgpt_bridge import sse_start, sse_stop


def download(url: str, path: Path):
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    path.write_bytes(response.content)
    logging.info(f"Downloaded {url} to {path}")


def wrap_sse(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        await sse_start()
        try:
            await f(*args, **kwargs)
        finally:
            await sse_stop()

    return decorated_function
