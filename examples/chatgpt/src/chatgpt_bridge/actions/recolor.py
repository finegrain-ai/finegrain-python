import asyncio

from finegrain import ErrorResult, StateID
from PIL import Image
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
    credits_left: int


async def process(
    ctx: EditorAPIContext,
    object_color: str,
    stateid_input: StateID,
    positive_prompt: str,
    negative_prompt: str | None,
) -> tuple[StateID, Image.Image]:
    # call multi_segment for positive prompt
    stateid_positive_segment = await ctx.call_async.multi_segment(
        state_id=stateid_input,
        prompt=positive_prompt,
    )
    app.logger.debug(f"{stateid_positive_segment=}")

    # call multi_segment for negative prompt
    if negative_prompt is not None:
        stateid_negative_segment = await ctx.call_async.multi_segment(
            state_id=stateid_input,
            prompt=negative_prompt,
        )
        app.logger.debug(f"{stateid_negative_segment=}")
    else:
        stateid_negative_segment = None

    # get difference between positive and negative masks
    if stateid_negative_segment is not None:
        result_mask_difference = await ctx.call_async.merge_masks(
            state_ids=[stateid_positive_segment, stateid_negative_segment],
            operation="difference",
        )
        if isinstance(result_mask_difference, ErrorResult):
            raise ValueError(f"[recolor] internal merge_masks error: {result_mask_difference.error}")
        stateid_mask_difference = result_mask_difference.state_id
    else:
        stateid_mask_difference = stateid_positive_segment
    app.logger.debug(f"{stateid_mask_difference=}")

    # recolor the image
    result_recolor = await ctx.call_async.recolor(
        image_state_id=stateid_input,
        mask_state_id=stateid_mask_difference,
        color=object_color,
    )
    if isinstance(result_recolor, ErrorResult):
        raise ValueError(f"[recolor] internal recolor error: {result_recolor.error}")
    stateid_recolor = result_recolor.state_id
    app.logger.debug(f"{stateid_recolor=}")

    # download output image
    pil_recolor = await ctx.call_async.download_pil_image(stateid_recolor)

    return stateid_recolor, pil_recolor


async def recolor(ctx: EditorAPIContext, request: Request) -> Response:
    # get information on the caller
    infos = await ctx.call_async.me()
    app.logger.debug(f"{infos['uid']} - {infos['credits']} - calling /recolor")

    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = RecolorParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # validate image input
    if input_data.stateids_input:
        len_stateids_input = len(input_data.stateids_input)
    elif input_data.openaiFileIdRefs:
        len_stateids_input = len(input_data.openaiFileIdRefs)
    else:
        raise ValueError("[recolor] input error: stateids_input or openaiFileIdRefs is required")

    # validate object_colors
    if input_data.object_colors is None:
        raise ValueError("[recolor] input error: object_colors is required")
    if len_stateids_input != len(input_data.object_colors):
        raise ValueError("[recolor] input error: stateids_input and object_colors must have the same length")

    # validate positive_prompts
    if input_data.positive_prompts is None:
        raise ValueError("[recolor] input error: positive_prompts is required")
    if len_stateids_input != len(input_data.positive_prompts):
        raise ValueError("[recolor] input error: stateids_input and positive_prompts must have the same length")
    if any(not prompt for prompt in input_data.positive_prompts):
        raise ValueError("[recolor] input error: all the positive prompts must be not empty")

    # validate negative_object_names
    if input_data.negative_prompts is None:
        input_data.negative_prompts = [None] * len_stateids_input
    if len_stateids_input != len(input_data.negative_prompts):
        raise ValueError("[recolor] input error: stateids_input and negative prompts must have the same length")

    # get stateids_input, or create them from openaiFileIdRefs
    if input_data.stateids_input:
        stateids_input = input_data.stateids_input
    elif input_data.openaiFileIdRefs:
        stateids_input = [await ref.get_stateid(ctx) for ref in input_data.openaiFileIdRefs]
    else:
        raise ValueError("[recolor] input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # process the inputs
    async with asyncio.TaskGroup() as tg:
        responses_recolor = [
            tg.create_task(
                process(
                    ctx=ctx,
                    object_color=object_color,
                    stateid_input=stateid_input,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                )
            )
            for object_color, stateid_input, positive_prompt, negative_prompt in zip(
                input_data.object_colors,
                stateids_input,
                input_data.positive_prompts,
                input_data.negative_prompts,
                strict=True,
            )
        ]
    results_recolor = [r.result() for r in responses_recolor]
    stateids_recolor = [r[0] for r in results_recolor]
    recolor_imgs = [r[1] for r in results_recolor]

    # get credits left
    infos = await ctx.call_async.me()
    app.logger.debug(f"{infos['uid']} - {infos['credits']} - done /recolor")

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
        credits_left=infos["credits"],
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
