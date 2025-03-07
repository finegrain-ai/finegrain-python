import base64
import io
from pathlib import Path

from PIL import Image
from quart.typing import TestClientProtocol

from chatgpt_bridge.skills.shadow import ShadowOutput, ShadowParams
from chatgpt_bridge.utils import OpenaiFileIdRef

from .utils import wrap_sse


@wrap_sse
async def test_shadow_no_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = ShadowParams(
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object_names is required"


@wrap_sse
async def test_shadow_empty_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = ShadowParams(
        object_names=[],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and object_names must have the same length"


@wrap_sse
async def test_shadow_empty_object_name(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = ShadowParams(
        object_names=[""],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object name cannot be empty"


@wrap_sse
async def test_shadow_no_images(
    test_client: TestClientProtocol,
) -> None:
    data = ShadowParams(
        object_names=["glass of water"],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input or openaiFileIdRefs is required"


@wrap_sse
async def test_shadow_empty_images(
    test_client: TestClientProtocol,
) -> None:
    data = ShadowParams(
        object_names=["glass of water"],
        openaiFileIdRefs=[],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input or openaiFileIdRefs is required"


@wrap_sse
async def test_shadow_object_names_wrong_cardinality(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = ShadowParams(
        object_names=["glass of water", "lamp"],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and object_names must have the same length"


@wrap_sse
async def test_shadow_empty_background_colors(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = ShadowParams(
        object_names=["glass of water"],
        openaiFileIdRefs=[example_ref],
        background_colors=[""],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "background color cannot be empty"


@wrap_sse
async def test_shadow_background_colors_wrong_cardinality(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = ShadowParams(
        object_names=["glass of water"],
        openaiFileIdRefs=[example_ref],
        background_colors=["#ff0000", "#ff0000"],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and background_colors must have the same length"


@wrap_sse
async def test_shadow(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
    tmp_path: Path,
) -> None:
    data = ShadowParams(
        object_names=["glass of water", "lamp"],
        openaiFileIdRefs=[example_ref, example_ref],
        background_colors=["#ff0000", "#00ff00"],
    )

    response = await test_client.post("/shadow", json=data.model_dump())
    response_json = await response.get_json()
    response_data = ShadowOutput(**response_json)

    assert response.status_code == 200
    assert len(response_json["openaiFileResponse"]) == 2
    for i, oai_file in enumerate(response_data.openaiFileResponse):
        assert oai_file.name == f"shadow_{i}.jpg"
        assert oai_file.mime_type == "image/jpeg"
        image_data = io.BytesIO(base64.b64decode(oai_file.content))
        image = Image.open(image_data)
        image.save(tmp_path / oai_file.name)
