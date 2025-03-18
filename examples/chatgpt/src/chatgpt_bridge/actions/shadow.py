from finegrain import ErrorResult, StateID
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse


class ShadowParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    background_colors: list[str] | None = None
    prompts: list[str] | None = None


class ShadowOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContext,
    stateid_input: StateID,
    prompt: str,
    background_color: str,
) -> StateID:
    # call detect
    result_detect = await ctx.call_async.detect(
        state_id=stateid_input,
        prompt=prompt,
    )
    if isinstance(result_detect, ErrorResult):
        raise ValueError(f"Shadow internal detect error: {result_detect.error}")
    detections = result_detect.results
    if len(detections) == 0:
        raise ValueError(f"Shadow internal detect error: no detection found for prompt {prompt}")
    app.logger.debug(f"{detections=}")

    # call segment on each detection
    stateids_segment = []
    for detection in detections:
        result_segment = await ctx.call_async.segment(
            state_id=stateid_input,
            bbox=detection.bbox,
        )
        if isinstance(result_segment, ErrorResult):
            raise ValueError(f"Shadow internal segment error: {result_segment.error}")
        stateids_segment.append(result_segment.state_id)
    app.logger.debug(f"{stateids_segment=}")

    # call merge-masks
    if len(stateids_segment) == 0:
        raise ValueError("Shadow internal merge_masks error: no segment found")
    elif len(stateids_segment) == 1:
        stateid_mask_union = stateids_segment[0]
    else:
        result_mask_union = await ctx.call_async.merge_masks(
            state_ids=tuple(stateids_segment),  # type: ignore
            operation="union",
        )
        if isinstance(result_mask_union, ErrorResult):
            raise ValueError(f"Shadow internal merge_masks error: {result_mask_union.error}")
        stateid_mask_union = result_mask_union.state_id
    app.logger.debug(f"{stateid_mask_union=}")

    # call cutout
    result_cutout = await ctx.call_async.cutout(
        image_state_id=stateid_input,
        mask_state_id=stateid_mask_union,
    )
    if isinstance(result_cutout, ErrorResult):
        raise ValueError(f"Shadow internal cutout error: {result_cutout.error}")
    stateid_cutout = result_cutout.state_id
    app.logger.debug(f"{stateid_cutout=}")

    # call shadow
    result_shadow = await ctx.call_async.shadow(
        state_id=stateid_cutout,
        background="transparent",
    )
    if isinstance(result_shadow, ErrorResult):
        raise ValueError(f"Shadow internal shadow error: {result_shadow.error}")
    stateid_shadow = result_shadow.state_id
    app.logger.debug(f"{stateid_shadow=}")

    # call set_background_color
    result_bgcolor = await ctx.call_async.set_background_color(
        state_id=stateid_shadow,
        background=background_color,
    )
    if isinstance(result_bgcolor, ErrorResult):
        raise ValueError(f"Shadow internal set_background_color error: {result_bgcolor.error}")
    stateid_bgcolor = result_bgcolor.state_id
    app.logger.debug(f"{stateid_bgcolor=}")

    return stateid_bgcolor


async def _shadow(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = ShadowParams(**input_json)
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
        raise ValueError("Shadow input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # validate prompts
    if input_data.prompts is None:
        raise ValueError("Shadow input error: prompts is required")
    if any(not prompt for prompt in input_data.prompts):
        raise ValueError("Shadow input error: all the prompts must be not empty")

    # validate background_colors
    if input_data.background_colors is None:
        input_data.background_colors = ["#ffffff"] * len(stateids_input)
    if len(stateids_input) != len(input_data.background_colors):
        raise ValueError("Shadow input error: stateids_input and background_colors must have the same length")
    if any(not color for color in input_data.background_colors):
        raise ValueError("Shadow input error: all the background colors must be not empty")

    # process the inputs
    stateids_shadow = [
        await process(
            ctx=ctx,
            stateid_input=stateid_input,
            prompt=prompt,
            background_color=background_color,
        )
        for stateid_input, prompt, background_color in zip(
            stateids_input,
            input_data.prompts,
            input_data.background_colors,
            strict=True,
        )
    ]

    # download output images
    shadow_imgs = [
        await ctx.call_async.download_pil_image(stateid_shadow)  #
        for stateid_shadow in stateids_shadow
    ]

    # build output response
    output_data = ShadowOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(image=shadow_img, name=f"shadow_{i}")
            for i, shadow_img in enumerate(shadow_imgs)
        ],
        stateids_output=stateids_shadow,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
