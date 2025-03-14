from finegrain import ErrorResult, StateID
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse


class RecolorParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    positive_object_names: list[list[str]] | None = None
    negative_object_names: list[list[str]] | None = None
    object_colors: list[str] | None = None


class RecolorOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContext,
    object_color: str,
    stateid_input: StateID,
    positive_object_names: list[str],
    negative_object_names: list[str],
) -> StateID:
    # get bounding boxes for positive objects
    stateids_bbox_positive = []
    for name in positive_object_names:
        result_bbox = await ctx.call_async.infer_bbox(
            state_id=stateid_input,
            product_name=name,
        )
        if isinstance(result_bbox, ErrorResult):
            raise ValueError(f"Recolor internal positive infer_bbox error: {result_bbox.error}")
        app.logger.debug(f"{result_bbox.bbox=}")
        stateids_bbox_positive.append(result_bbox.bbox)
    app.logger.debug(f"{stateids_bbox_positive=}")

    # get bounding boxes for negative objects
    stateids_bbox_negative = []
    for name in negative_object_names:
        result_bbox = await ctx.call_async.infer_bbox(
            state_id=stateid_input,
            product_name=name,
        )
        if isinstance(result_bbox, ErrorResult):
            raise ValueError(f"Recolor internal negative infer_bbox error: {result_bbox.error}")
        app.logger.debug(f"{result_bbox.bbox=}")
        stateids_bbox_negative.append(result_bbox.bbox)
    app.logger.debug(f"{stateids_bbox_negative=}")

    # get segments for positive objects
    stateids_mask_positive = []
    for bbox in stateids_bbox_positive:
        result_segment = await ctx.call_async.segment(
            state_id=stateid_input,
            bbox=bbox,
        )
        if isinstance(result_segment, ErrorResult):
            raise ValueError(f"Recolor internal positive segment error: {result_segment.error}")
        stateids_mask_positive.append(result_segment.state_id)
    app.logger.debug(f"{stateids_mask_positive=}")

    # get segments for negative objects
    stateids_mask_negative = []
    for bbox in stateids_bbox_negative:
        result_segment = await ctx.call_async.segment(
            state_id=stateid_input,
            bbox=bbox,
        )
        if isinstance(result_segment, ErrorResult):
            raise ValueError(f"Recolor internal negative segment error: {result_segment.error}")
        stateids_mask_negative.append(result_segment.state_id)
    app.logger.debug(f"{stateids_mask_negative=}")

    # merge positive masks
    if len(stateids_mask_positive) == 1:
        stateid_mask_positive_union = stateids_mask_positive[0]
    else:
        result_mask_union = await ctx.call_async.merge_masks(
            state_ids=tuple(stateids_mask_positive),  # type: ignore
            operation="union",
        )
        if isinstance(result_mask_union, ErrorResult):
            raise ValueError(f"Recolor internal positive merge_masks error: {result_mask_union.error}")
        stateid_mask_positive_union = result_mask_union.state_id
    app.logger.debug(f"{stateid_mask_positive_union=}")

    # merge negative masks
    if len(stateids_mask_negative) == 0:
        stateid_mask_negative_union = None
    elif len(stateids_mask_negative) == 1:
        stateid_mask_negative_union = stateids_mask_negative[0]
    else:
        result_mask_union = await ctx.call_async.merge_masks(
            state_ids=tuple(stateids_mask_negative),  # type: ignore
            operation="union",
        )
        if isinstance(result_mask_union, ErrorResult):
            raise ValueError(f"Recolor internal negative merge_masks error: {result_mask_union.error}")
        stateid_mask_negative_union = result_mask_union.state_id
    app.logger.debug(f"{stateid_mask_negative_union=}")

    # get difference between positive and negative masks
    if stateid_mask_negative_union is not None:
        result_mask_difference = await ctx.call_async.merge_masks(
            state_ids=(stateid_mask_positive_union, stateid_mask_negative_union),  # type: ignore
            operation="difference",
        )
        if isinstance(result_mask_difference, ErrorResult):
            raise ValueError(f"Recolor internal merge_masks error: {result_mask_difference.error}")
        stateid_mask_difference = result_mask_difference.state_id
    else:
        stateid_mask_difference = stateid_mask_positive_union
    app.logger.debug(f"{stateid_mask_difference=}")

    # recolor the image
    result_recolor = await ctx.call_async.recolor(
        image_state_id=stateid_input,
        mask_state_id=stateid_mask_difference,
        color=object_color,
    )
    if isinstance(result_recolor, ErrorResult):
        raise ValueError(f"Recolor internal recolor error: {result_recolor.error}")
    stateid_recolor = result_recolor.state_id
    app.logger.debug(f"{stateid_recolor=}")

    return stateid_recolor


async def _recolor(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = RecolorParams(**input_json)
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
        raise ValueError("Recolor input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # validate object_colors
    if input_data.object_colors is None:
        raise ValueError("Recolor input error: object_colors is required")
    if len(stateids_input) != len(input_data.object_colors):
        raise ValueError("Recolor input error: stateids_input and object_colors must have the same length")

    # validate positive_object_names
    if input_data.positive_object_names is None:
        raise ValueError("Recolor input error: positive_object_names is required")
    if len(stateids_input) != len(input_data.positive_object_names):
        raise ValueError("Recolor input error: stateids_input and positive_object_names must have the same length")
    for object_names in input_data.positive_object_names:
        if not object_names:
            raise ValueError("Recolor input error: positive object list cannot be empty")
        for object_name in object_names:
            if not object_name:
                raise ValueError("Recolor input error: positive object name cannot be empty")

    # validate negative_object_names
    if input_data.negative_object_names is None:
        input_data.negative_object_names = [[]] * len(stateids_input)
    if len(stateids_input) != len(input_data.negative_object_names):
        raise ValueError("Recolor input error: stateids_input and negative_object_names must have the same length")
    for object_names in input_data.negative_object_names:
        for object_name in object_names:
            if not object_name:
                raise ValueError("Recolor input error: negative object name cannot be empty")

    # process the inputs
    stateids_recolor = [
        await process(
            ctx=ctx,
            object_color=object_color,
            stateid_input=stateid_input,
            positive_object_names=positive_object_names,
            negative_object_names=negative_object_names,
        )
        for stateid_input, positive_object_names, negative_object_names, object_color in zip(
            stateids_input,
            input_data.positive_object_names,
            input_data.negative_object_names,
            input_data.object_colors,
            strict=True,
        )
    ]
    app.logger.debug(f"{stateids_recolor=}")

    # download output images
    recolor_imgs = [
        await ctx.call_async.download_pil_image(stateid_recolor)  #
        for stateid_recolor in stateids_recolor
    ]

    # build output response
    output_data = RecolorOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(
                image=recolor_img,
                name=f"recolored_{i}",
            )
            for i, recolor_img in enumerate(recolor_imgs)
        ],
        stateids_output=stateids_recolor,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
