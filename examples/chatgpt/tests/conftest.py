from functools import partial

import pytest
from quart.typing import TestClientProtocol

from chatgpt_bridge import app, login
from chatgpt_bridge.env import CHATGPT_AUTH_TOKEN
from chatgpt_bridge.utils import OpenaiFileIdRef


@pytest.fixture()
async def test_client() -> TestClientProtocol:
    await login()
    client = app.test_client()
    client.post = partial(client.post, headers={"Authorization": f"Basic {CHATGPT_AUTH_TOKEN}"})
    return client


@pytest.fixture()
def example_ref() -> OpenaiFileIdRef:
    return OpenaiFileIdRef(
        name="image.jpg",
        id="file-AAAAAAAAAAAAAAAAAAAAAAAA",
        mime_type="image/jpeg",
        download_link="https://img.freepik.com/free-photo/still-life-device-table_23-2150994394.jpg",
    )
