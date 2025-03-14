from finegrain import BoundingBox, ErrorResult, StateID
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse


class EraseParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    object_names: list[list[str]] | None = None


class EraseOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContext,
    stateid_input: StateID,
    object_names: list[str],
) -> StateID:
    # call infer-bbox for each object
    bboxes: list[BoundingBox] = []
    for name in object_names:
        result_bbox = await ctx.call_async.infer_bbox(
            state_id=stateid_input,
            product_name=name,
        )
        if isinstance(result_bbox, ErrorResult):
            raise ValueError(f"Eraser internal infer_bbox error: {result_bbox.error}")
        bboxes.append(result_bbox.bbox)
    app.logger.debug(f"{bboxes=}")

    # call segment for each bbox
    stateids_segment = []
    for result_bbox in bboxes:
        result_segment = await ctx.call_async.segment(
            state_id=stateid_input,
            bbox=result_bbox,
        )
        if isinstance(result_segment, ErrorResult):
            raise ValueError(f"Erase internal segment error: {result_segment.error}")
        stateids_segment.append(result_segment.state_id)
    app.logger.debug(f"{stateids_segment=}")

    # call merge-masks
    if len(stateids_segment) == 1:
        stateid_mask_union = stateids_segment[0]
    else:
        result_mask_union = await ctx.call_async.merge_masks(
            state_ids=tuple(stateids_segment),  # type: ignore
            operation="union",
        )
        if isinstance(result_mask_union, ErrorResult):
            raise ValueError(f"Eraser internal merge_masks error: {result_mask_union.error}")
        stateid_mask_union = result_mask_union.state_id
    app.logger.debug(f"{stateid_mask_union=}")

    # call erase skill
    result_erase = await ctx.call_async.erase(
        image_state_id=stateid_input,
        mask_state_id=stateid_mask_union,
        mode="express",
    )
    if isinstance(result_erase, ErrorResult):
        raise ValueError(f"Eraser internal erase error: {result_erase.error}")
    stateid_erase = result_erase.state_id
    app.logger.debug(f"{stateid_erase=}")

    return stateid_erase


async def _eraser(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = EraseParams(**input_json)
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
        raise ValueError("Eraser input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # validate the inputs
    if input_data.object_names is None:
        raise ValueError("Eraser input error: object_names is required")
    if len(stateids_input) != len(input_data.object_names):
        raise ValueError("Eraser input error: stateids_input and object_names must have the same length")
    for object_names in input_data.object_names:
        if not object_names:
            raise ValueError("Eraser input error: object list cannot be empty")
        for object_name in object_names:
            if not object_name:
                raise ValueError("Eraser input error: object name cannot be empty")

    # process the inputs
    stateids_erased = [
        await process(ctx, stateid_input, object_names)
        for stateid_input, object_names in zip(stateids_input, input_data.object_names, strict=True)
    ]
    app.logger.debug(f"{stateids_erased=}")

    # download images from API
    pil_outputs = [
        await ctx.call_async.download_pil_image(stateid_erased_img)  #
        for stateid_erased_img in stateids_erased
    ]

    # build output response
    data_output = EraseOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(image=erased_img, name=f"erased_{i}")
            for i, erased_img in enumerate(pil_outputs)
        ],
        stateids_output=stateids_erased,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{data_output=}")
    output_response = jsonify(data_output.model_dump())
    return output_response
