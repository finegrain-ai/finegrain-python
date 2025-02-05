import base64
import io
from functools import wraps

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
