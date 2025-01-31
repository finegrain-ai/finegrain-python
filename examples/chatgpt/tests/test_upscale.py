import base64
import io
from pathlib import Path

from PIL import Image
from quart.typing import TestClientProtocol

from chatgpt_bridge.skills.upscale import UpscaleOutput, UpscaleParams
from chatgpt_bridge.utils import OpenaiFileIdRef

from .utils import wrap_sse


@wrap_sse
async def test_image_upscale_no_images(test_client: TestClientProtocol) -> None:
    data = UpscaleParams()

    response = await test_client.post("/upscale", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input or openaiFileIdRefs is required"


@wrap_sse
async def test_image_upscale_empty_images(test_client: TestClientProtocol) -> None:
    data = UpscaleParams(
        openaiFileIdRefs=[],
    )

    response = await test_client.post("/upscale", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input or openaiFileIdRefs is required"


@wrap_sse
async def test_image_upscale(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
    tmp_path: Path,
) -> None:
    data = UpscaleParams(
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/upscale", json=data.model_dump())
    response_json = await response.get_json()
    response_data = UpscaleOutput(**response_json)

    assert response.status_code == 200
    assert response.status_code == 200
    assert len(response_json["openaiFileResponse"]) == 1
    for i, oai_file in enumerate(response_data.openaiFileResponse):
        assert oai_file.name == f"upscaled_{i}.jpg"
        assert oai_file.mime_type == "image/jpeg"
        image_data = io.BytesIO(base64.b64decode(oai_file.content))
        image = Image.open(image_data)
        image.save(tmp_path / oai_file.name)
