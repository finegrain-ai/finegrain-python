from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContextCached
from chatgpt_bridge.utils import BoundingBox, OpenaiFileIdRef, StateID, json_error


class BoxParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    object_names: list[str] | None = None


class BoxOutput(BaseModel):
    bounding_boxes: list[BoundingBox]


async def process(
    ctx: EditorAPIContextCached,
    stateid_input: StateID,
    object_name: str,
) -> BoundingBox:
    # queue skills/infer-bbox
    stateid_bbox = await ctx.skill_bbox(stateid_image=stateid_input, product_name=object_name)
    app.logger.debug(f"{stateid_bbox=}")

    # get bbox state/meta
    metadata_bbox = await ctx.get_meta(stateid_bbox)
    bounding_box = metadata_bbox["bbox"]

    return bounding_box


async def _box(ctx: EditorAPIContextCached, request: Request) -> Response:
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
                stateid_input = await ctx.create_state(oai_ref.download_link)
                stateids_input.append(stateid_input)
    else:
        return json_error("stateids_input or openaiFileIdRefs is required", 400)
    app.logger.debug(f"{stateids_input=}")

    # validate object_names
    if input_data.object_names is None:
        return json_error("object_names is required", 400)
    if not input_data.object_names:
        return json_error("object_names cannot be empty", 400)
    for object_name in input_data.object_names:
        if not object_name:
            return json_error("object name cannot be empty", 400)
    if len(input_data.object_names) != len(stateids_input):
        return json_error("stateids_input and object_names must have the same length", 400)

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
