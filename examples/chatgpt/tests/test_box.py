from quart.typing import TestClientProtocol

from chatgpt_bridge.skills.box import BoxParams
from chatgpt_bridge.utils import OpenaiFileIdRef

from .utils import wrap_sse


@wrap_sse
async def test_box_no_object_name(test_client: TestClientProtocol, example_ref: OpenaiFileIdRef) -> None:
    data = BoxParams(
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/box", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object_names is required"


@wrap_sse
async def test_box_no_images(test_client: TestClientProtocol) -> None:
    data = BoxParams(
        object_names=["glass of water"],
    )

    response = await test_client.post("/box", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input or openaiFileIdRefs is required"


@wrap_sse
async def test_box_empty_object_names(test_client: TestClientProtocol, example_ref: OpenaiFileIdRef) -> None:
    data = BoxParams(
        object_names=[],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/box", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object_names cannot be empty"


@wrap_sse
async def test_box_empty_object_name(test_client: TestClientProtocol, example_ref: OpenaiFileIdRef) -> None:
    data = BoxParams(
        object_names=[""],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/box", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "object name cannot be empty"


@wrap_sse
async def test_box_object_names_wrong_cardinality(
    test_client: TestClientProtocol,
    example_ref: OpenaiFileIdRef,
) -> None:
    data = BoxParams(
        object_names=["glass of water", "lamp"],
        openaiFileIdRefs=[example_ref],
    )

    response = await test_client.post("/box", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 400
    assert response_json["error"] == "stateids_input and object_names must have the same length"


@wrap_sse
async def test_box(test_client: TestClientProtocol, example_ref: OpenaiFileIdRef) -> None:
    data = BoxParams(
        object_names=["glass of water", "lamp"],
        openaiFileIdRefs=[example_ref, example_ref],
    )

    response = await test_client.post("/box", json=data.model_dump())
    response_json = await response.get_json()

    assert response.status_code == 200
    assert "bounding_boxes" in response_json
    bounding_boxes = response_json["bounding_boxes"]
    assert len(bounding_boxes) == 2
