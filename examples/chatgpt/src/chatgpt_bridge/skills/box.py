from finegrain import BoundingBox, ErrorResult, StateID
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileIdRef


class BoxParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    object_names: list[str] | None = None


class BoxOutput(BaseModel):
    bounding_boxes: list[BoundingBox]


async def process(
    ctx: EditorAPIContext,
    stateid_input: StateID,
    object_name: str,
) -> BoundingBox:
    # call infer-bbox
    result_bbox = await ctx.call_async.infer_bbox(
        state_id=stateid_input,
        product_name=object_name,
    )
    if isinstance(result_bbox, ErrorResult):
        raise ValueError(f"Box internal infer_bbox error: {result_bbox.error}")
    bbox = result_bbox.bbox
    app.logger.debug(f"{bbox=}")

    return bbox


async def _box(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = BoxParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # get stateids_input, or create them from openaiFileIdRefs
    if input_data.stateids_input:
        stateids_input = input_data.stateids_input
    elif input_data.openaiFileIdRefs:
        stateids_input: list[StateID] = []
        for oai_ref in input_data.openaiFileIdRefs:
            if oai_ref.download_link:
                stateid_input = await ctx.call_async.upload_link_image(oai_ref.download_link)
                stateids_input.append(stateid_input)
    else:
        raise ValueError("Box input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # validate object_names
    if input_data.object_names is None:
        raise ValueError("Box input error: object_names is required")
    if not input_data.object_names:
        raise ValueError("Box input error: object_names cannot be empty")
    for object_name in input_data.object_names:
        if not object_name:
            raise ValueError("Box input error: object name cannot be empty")
    if len(input_data.object_names) != len(stateids_input):
        raise ValueError("Box input error: stateids_input and object_names must have the same length")

    # process the inputs
    bounding_boxes = [
        await process(ctx, stateid_input, object_name)
        for stateid_input, object_name in zip(
            stateids_input,
            input_data.object_names,
            strict=True,
        )
    ]

    # build output response
    output_data = BoxOutput(bounding_boxes=bounding_boxes)
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
