from finegrain import ErrorResult, StateID
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse


class RecolorParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    positive_prompts: list[str] | None = None
    negative_prompts: list[str] | list[None] | None = None
    object_colors: list[str] | None = None


class RecolorOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContext,
    object_color: str,
    stateid_input: StateID,
    positive_prompt: str,
    negative_prompt: str | None,
) -> StateID:
    # call detect for positive objects
    result_positive_detect = await ctx.call_async.detect(
        state_id=stateid_input,
        prompt=positive_prompt,
    )
    if isinstance(result_positive_detect, ErrorResult):
        raise ValueError(f"Recolor internal positive detect error: {result_positive_detect.error}")
    positive_detections = result_positive_detect.results
    if len(positive_detections) == 0:
        raise ValueError(f"Recolor internal positive detect error: no detection found for prompt {positive_prompt}")
    app.logger.debug(f"{positive_detections=}")

    # call detect for negative objects
    negative_detections = []
    if negative_prompt:
        result_negative_detect = await ctx.call_async.detect(
            state_id=stateid_input,
            prompt=negative_prompt,
        )
        if isinstance(result_negative_detect, ErrorResult):
            raise ValueError(f"Recolor internal negative detect error: {result_negative_detect.error}")
        negative_detections = result_negative_detect.results
    app.logger.debug(f"{negative_detections=}")

    # get segments for positive objects
    stateids_mask_positive = []
    for detection in positive_detections:
        result_segment = await ctx.call_async.segment(
            state_id=stateid_input,
            bbox=detection.bbox,
        )
        if isinstance(result_segment, ErrorResult):
            raise ValueError(f"Recolor internal positive segment error: {result_segment.error}")
        stateids_mask_positive.append(result_segment.state_id)
    app.logger.debug(f"{stateids_mask_positive=}")

    # get segments for negative objects
    stateids_mask_negative = []
    if negative_detections:
        for detection in negative_detections:
            result_segment = await ctx.call_async.segment(
                state_id=stateid_input,
                bbox=detection.bbox,
            )
            if isinstance(result_segment, ErrorResult):
                raise ValueError(f"Recolor internal negative segment error: {result_segment.error}")
            stateids_mask_negative.append(result_segment.state_id)
    app.logger.debug(f"{stateids_mask_negative=}")

    # merge positive masks
    if len(stateids_mask_positive) == 0:
        raise ValueError("Recolor internal positive merge masks error: no segment found")
    elif len(stateids_mask_positive) == 1:
        stateid_mask_positive_union = stateids_mask_positive[0]
    else:
        result_mask_union = await ctx.call_async.merge_masks(
            state_ids=stateids_mask_positive,
            operation="union",
        )
        if isinstance(result_mask_union, ErrorResult):
            raise ValueError(f"Recolor internal positive merge masks error: {result_mask_union.error}")
        stateid_mask_positive_union = result_mask_union.state_id
    app.logger.debug(f"{stateid_mask_positive_union=}")

    # merge negative masks
    if len(stateids_mask_negative) == 0:
        stateid_mask_negative_union = None
    elif len(stateids_mask_negative) == 1:
        stateid_mask_negative_union = stateids_mask_negative[0]
    else:
        result_mask_union = await ctx.call_async.merge_masks(
            state_ids=stateids_mask_negative,
            operation="union",
        )
        if isinstance(result_mask_union, ErrorResult):
            raise ValueError(f"Recolor internal negative merge masks error: {result_mask_union.error}")
        stateid_mask_negative_union = result_mask_union.state_id
    app.logger.debug(f"{stateid_mask_negative_union=}")

    # get difference between positive and negative masks
    if stateid_mask_negative_union is not None:
        result_mask_difference = await ctx.call_async.merge_masks(
            state_ids=[stateid_mask_positive_union, stateid_mask_negative_union],
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

    # validate positive_prompts
    if input_data.positive_prompts is None:
        raise ValueError("Recolor input error: positive_prompts is required")
    if len(stateids_input) != len(input_data.positive_prompts):
        raise ValueError("Recolor input error: stateids_input and positive_prompts must have the same length")
    if any(not prompt for prompt in input_data.positive_prompts):
        raise ValueError("Recolor input error: all the positive prompts must be not empty")

    # validate negative_object_names
    if input_data.negative_prompts is None:
        input_data.negative_prompts = [None] * len(stateids_input)
    if len(stateids_input) != len(input_data.negative_prompts):
        raise ValueError("Recolor input error: stateids_input and negative prompts must have the same length")

    # process the inputs
    stateids_recolor = [
        await process(
            ctx=ctx,
            object_color=object_color,
            stateid_input=stateid_input,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
        )
        for stateid_input, positive_prompt, negative_prompt, object_color in zip(
            stateids_input,
            input_data.positive_prompts,
            input_data.negative_prompts,
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
