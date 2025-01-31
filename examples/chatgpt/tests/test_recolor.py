import base64
import io
import logging
from pathlib import Path

from PIL import Image
from quart.typing import TestClientProtocol

from chatgpt_bridge.skills.recolor import RecolorOutput, RecolorParams
from chatgpt_bridge.utils import OpenaiFileIdRef

from .utils import wrap_sse


@wrap_sse
async def test_recolor_no_positive_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        object_colors=["#ff0000"],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "positive_object_names is required"


@wrap_sse
async def test_recolor_no_object_colors(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        positive_object_names=[["glass of water"]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object_colors is required"


@wrap_sse
async def test_recolor_no_images(
    test_client: TestClientProtocol,
) -> None:
    data = RecolorParams(
        object_colors=["#ff0000"],
        positive_object_names=[["glass of water"]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input or openaiFileIdRefs is required"


@wrap_sse
async def test_recolor_empty_positive_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        object_colors=["#ff0000"],
        positive_object_names=[[]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "positive object list cannot be empty"


@wrap_sse
async def test_recolor_empty_negative_object_name(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        object_colors=["#ff0000"],
        positive_object_names=[["glass of water"]],
        negative_object_names=[[""]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "negative object name cannot be empty"


@wrap_sse
async def test_recolor_empty2_positive_object_names(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        object_colors=["#ff0000"],
        positive_object_names=[[""]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "positive object name cannot be empty"


@wrap_sse
async def test_recolor_wrong_positive_object_color_cardinality(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        object_colors=["#ff0000", "#ff0000"],
        positive_object_names=[["glass of water"]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and object_colors must have the same length"


@wrap_sse
async def test_recolor_wrong_positive_object_names_cardinality(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        object_colors=["#ff0000"],
        positive_object_names=[["glass of water"], ["glass of water"]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and positive_object_names must have the same length"


@wrap_sse
async def test_recolor_wrong_negative_object_names_cardinality(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref],
        object_colors=["#ff0000"],
        positive_object_names=[["glass of water"]],
        negative_object_names=[["bowl"], ["bowl"]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and negative_object_names must have the same length"


@wrap_sse
async def test_recolor(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
    tmp_path: Path,
) -> None:
    data = RecolorParams(
        openaiFileIdRefs=[example_ref, example_ref, example_ref],
        object_colors=["#ff0000", "#00ff00", "#0000ff"],
        positive_object_names=[["glass of water"], ["bowl", "glass of water"], ["bowl", "glass of water"]],
        negative_object_names=[[], [], ["bowl"]],
    )

    response = await test_client.post("/recolor", json=data.model_dump())
    response_json = await response.get_json()
    response_data = RecolorOutput(**response_json)

    assert response.status_code == 200
    assert len(response_json["openaiFileResponse"]) == 3
    for i, oai_file in enumerate(response_data.openaiFileResponse):
        assert oai_file.name == f"recolored_{i}.jpg"
        assert oai_file.mime_type == "image/jpeg"
        image_data = io.BytesIO(base64.b64decode(oai_file.content))
        image = Image.open(image_data)
        image.save(tmp_path / oai_file.name)
        logging.info(f"Saved image to {tmp_path / oai_file.name}")
