import base64
import io
from pathlib import Path

from PIL import Image
from quart.typing import TestClientProtocol

from chatgpt_bridge.skills.erase import EraseOutput, EraseParams
from chatgpt_bridge.utils import OpenaiFileIdRef
from tests.utils import wrap_sse


@wrap_sse
async def test_eraser_no_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = EraseParams(
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/erase", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object_names is required"


@wrap_sse
async def test_eraser_empty_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = EraseParams(
        object_names=[],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/erase", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and object_names must have the same length"


@wrap_sse
async def test_eraser_empty2_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = EraseParams(
        object_names=[[]],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/erase", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object list cannot be empty"


@wrap_sse
async def test_eraser_empty_object_name(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = EraseParams(
        object_names=[[""]],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/erase", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object name cannot be empty"


@wrap_sse
async def test_eraser_object_names_wrong_cardinality(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = EraseParams(
        object_names=[["glass of water"]],
        openaiFileIdRefs=[example_ref, example_ref],
    )

    response = await test_client.post("/erase", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and object_names must have the same length"


@wrap_sse
async def test_eraser_no_images(test_client: TestClientProtocol) -> None:
    data = EraseParams(
        object_names=[["glass of water"]],
        openaiFileIdRefs=[],
    )

    response = await test_client.post("/erase", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input or openaiFileIdRefs is required"


@wrap_sse
async def test_eraser(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
    tmp_path: Path,
) -> None:
    data = EraseParams(
        object_names=[["glass of water"], ["glass of water", "lamp"]],
        openaiFileIdRefs=[example_ref, example_ref],
    )

    response = await test_client.post("/erase", json=data.model_dump())
    response_json = await response.get_json()
    response_data = EraseOutput(**response_json)

    assert response.status_code == 200
    assert len(response_json["openaiFileResponse"]) == 2
    for i, oai_file in enumerate(response_data.openaiFileResponse):
        assert oai_file.name == f"erased_{i}.jpg"
        assert oai_file.mime_type == "image/jpeg"
        image_data = io.BytesIO(base64.b64decode(oai_file.content))
        image = Image.open(image_data)
        image.save(tmp_path / oai_file.name)
