import base64
import io
from functools import wraps

from finegrain import EditorAPIContext
from PIL import Image
from pydantic import BaseModel
from quart import Response, jsonify, request
from quart import current_app as app

StateID = str


def json_error(message: str, status: int = 400) -> Response:
    response = jsonify(error=message)
    response.status_code = status
    app.logger.error(message)
    return response


def require_basic_auth_token(token: str):
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Basic {token}":
                return json_error("Unauthorized", 401)
            return await f(*args, **kwargs)

        return decorated_function

    return decorator


async def create_state(
    ctx: EditorAPIContext,
    file_url: str | None = None,
    file: io.BytesIO | None = None,
    timeout: float | None = None,
) -> str:
    assert file_url or file, "file_url or file is required"

    response = await ctx.request(
        method="POST",
        url="state/create",
        json={"priority": ctx.priority},
        data={"file_url": file_url} if file_url else None,
        files={"file": file} if file else None,
    )
    state_id = response.json()["state"]
    status = await ctx.sse_await(state_id, timeout=timeout)
    if status:
        return state_id
    meta = await ctx.get_meta(state_id)
    raise RuntimeError(f"create_state failed with {state_id}: {meta}")


async def download_image(
    ctx: EditorAPIContext,
    stateid: str,
    image_format: str = "PNG",
    image_resolution: str = "DISPLAY",
) -> Image.Image:
    response = await ctx.request(
        method="GET",
        url=f"state/image/{stateid}",
        params={
            "format": image_format,
            "resolution": image_resolution,
        },
    )
    return Image.open(io.BytesIO(response.content))


def image_to_base64(
    image: Image.Image,
    image_format: str,
) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OpenaiFileIdRef(BaseModel):
    # https://platform.openai.com/docs/actions/sending-files
    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    download_link: str


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
            content=image_to_base64(image.convert("RGB"), "JPEG"),
        )

    def __repr__(self) -> str:
        return f"OpenaiFileResponse(name={self.name}, mime_type={self.mime_type}, content_len={len(self.content)})"


BoundingBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)
