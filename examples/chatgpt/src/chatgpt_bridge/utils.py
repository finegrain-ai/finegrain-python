import base64
import io
from contextlib import asynccontextmanager

from finegrain import StateID
from PIL import Image
from pydantic import BaseModel
from quart import Response, jsonify, request
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.env import (
    FG_API_PRIORITY,
    FG_API_TIMEOUT,
    FG_API_URL,
    USER_AGENT,
)

FG_API_DUMMY_KEY = "FGAPI-DUMMMY-DUMMMY-DUMMMY-DUMMMY"


def json_error(message: str | list[str], status: int = 400) -> Response:
    response = jsonify(error=message)
    response.status_code = status
    app.logger.error(message)
    return response


@asynccontextmanager
async def get_ctx():
    # parse the token from the Authorization header
    token = request.headers.get("Authorization")
    assert token is not None, "Authorization header is required"
    assert token.startswith("Bearer "), "Authorization token must be a Bearer token"
    token = token.removeprefix("Bearer ")

    # create a new context
    ctx = EditorAPIContext(
        api_key=FG_API_DUMMY_KEY,
        base_url=FG_API_URL,
        priority=FG_API_PRIORITY,
        default_timeout=FG_API_TIMEOUT,
        user_agent=USER_AGENT,
    )

    # set the token
    ctx.token = token

    # start the sse loop
    await ctx.sse_start()

    # yield the context
    yield ctx

    # stop the sse loop
    await ctx.sse_stop()

    # clear the token
    ctx.token = None


def image_to_bytes(image: Image.Image) -> io.BytesIO:
    data = io.BytesIO()
    image.convert("RGB").save(data, format="JPEG", quality=95)
    data.seek(0)
    return data


def image_to_base64(image: Image.Image) -> str:
    image_data = image_to_bytes(image).getvalue()
    return base64.b64encode(image_data).decode("utf-8")


class OpenaiFileIdRef(BaseModel):
    # https://platform.openai.com/docs/actions/sending-files
    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    download_link: str

    async def get_stateid(self, ctx: EditorAPIContext) -> StateID:
        return await ctx.call_async.upload_link_image(self.download_link)


class OpenaiFileResponse(BaseModel):
    # https://platform.openai.com/docs/actions/sending-files
    name: str
    mime_type: str
    content: str

    @staticmethod
    def from_image(image: Image.Image, name: str) -> "OpenaiFileResponse":
        return OpenaiFileResponse(
            name=f"{name}.jpg",
            mime_type="image/jpeg",
            content=image_to_base64(image),
        )

    def __repr__(self) -> str:
        return f"OpenaiFileResponse(name={self.name}, mime_type={self.mime_type}, content_len={len(self.content)})"
