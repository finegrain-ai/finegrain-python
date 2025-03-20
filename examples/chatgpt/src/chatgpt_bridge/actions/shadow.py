import asyncio

from finegrain import ErrorResult, StateID
from PIL import Image
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
) -> tuple[StateID, Image.Image]:
    # call multi_segment
    stateid_segment = await ctx.call_async.multi_segment(
        state_id=stateid_input,
        prompt=prompt,
    )
    app.logger.debug(f"{stateid_segment=}")

    # call cutout
    result_cutout = await ctx.call_async.cutout(
        image_state_id=stateid_input,
        mask_state_id=stateid_segment,
    )
    if isinstance(result_cutout, ErrorResult):
        raise ValueError(f"[shadow] internal cutout error: {result_cutout.error}")
    stateid_cutout = result_cutout.state_id
    app.logger.debug(f"{stateid_cutout=}")

    # call shadow
    result_shadow = await ctx.call_async.shadow(
        state_id=stateid_cutout,
        background="transparent",
    )
    if isinstance(result_shadow, ErrorResult):
        raise ValueError(f"[shadow] internal shadow error: {result_shadow.error}")
    stateid_shadow = result_shadow.state_id
    app.logger.debug(f"{stateid_shadow=}")

    # call set_background_color
    result_bgcolor = await ctx.call_async.set_background_color(
        state_id=stateid_shadow,
        background=background_color,
    )
    if isinstance(result_bgcolor, ErrorResult):
        raise ValueError(f"[shadow] internal set_background_color error: {result_bgcolor.error}")
    stateid_bgcolor = result_bgcolor.state_id
    app.logger.debug(f"{stateid_bgcolor=}")

    # download output image
    pil_bgcolor = await ctx.call_async.download_pil_image(stateid_bgcolor)

    return stateid_bgcolor, pil_bgcolor


async def shadow(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = ShadowParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # validate image input
    if input_data.stateids_input:
        len_stateids_input = len(input_data.stateids_input)
    elif input_data.openaiFileIdRefs:
        len_stateids_input = len(input_data.openaiFileIdRefs)
    else:
        raise ValueError("[shadow] input error: stateids_input or openaiFileIdRefs is required")

    # validate prompts
    if input_data.prompts is None:
        raise ValueError("[shadow] input error: prompts is required")
    if any(not prompt for prompt in input_data.prompts):
        raise ValueError("[shadow] input error: all the prompts must be not empty")

    # validate background_colors
    if input_data.background_colors is None:
        input_data.background_colors = ["#ffffff"] * len_stateids_input
    if len_stateids_input != len(input_data.background_colors):
        raise ValueError("[shadow] input error: stateids_input and background_colors must have the same length")
    if any(not color for color in input_data.background_colors):
        raise ValueError("[shadow] input error: all the background colors must be not empty")

    # get stateids_input, or create them from openaiFileIdRefs
    if input_data.stateids_input:
        stateids_input = input_data.stateids_input
    elif input_data.openaiFileIdRefs:
        stateids_input = [await ref.get_stateid(ctx) for ref in input_data.openaiFileIdRefs]
    else:
        raise ValueError("[shadow] input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # process the inputs
    async with asyncio.TaskGroup() as tg:
        responses_shadow = [
            tg.create_task(
                process(
                    ctx=ctx,
                    stateid_input=stateid_input,
                    prompt=prompt,
                    background_color=background_color,
                )
            )
            for stateid_input, prompt, background_color in zip(
                stateids_input,
                input_data.prompts,
                input_data.background_colors,
                strict=True,
            )
        ]
    results_shadow = [r.result() for r in responses_shadow]
    stateids_shadow = [r[0] for r in results_shadow]
    pils_shadow = [r[1] for r in results_shadow]

    # build output response
    output_data = ShadowOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(image=shadow_img, name=f"shadow_{i}")
            for i, shadow_img in enumerate(pils_shadow)
        ],
        stateids_output=stateids_shadow,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
