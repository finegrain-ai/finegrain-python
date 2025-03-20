import asyncio

from finegrain import ErrorResult, StateID
from PIL import Image
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse, image_to_bytes


class CutoutParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    background_colors: list[str] | None = None
    prompts: list[str] | None = None


class CutoutOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_undo: list[StateID]
    stateids_output: list[StateID]
    credits_left: int


async def process(
    ctx: EditorAPIContext,
    stateid_input: StateID,
    background_color: str,
    prompt: str,
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
        raise ValueError(f"[cutout] internal cutout error: {result_cutout.error}")
    stateid_cutout = result_cutout.state_id
    app.logger.debug(f"{stateid_cutout=}")

    # download cutout
    pil_cutout = await ctx.call_async.download_pil_image(stateid_cutout)

    # paste cutout onto a blank image, with margins
    pil_cutout_margin = Image.new(
        mode="RGBA",
        size=(int(1.618 * pil_cutout.width), int(1.618 * pil_cutout.height)),
        color=background_color,
    )
    bbox = (
        (pil_cutout_margin.width - pil_cutout.width) // 2,
        (pil_cutout_margin.height - pil_cutout.height) // 2,
        (pil_cutout_margin.width - pil_cutout.width) // 2 + pil_cutout.width,
        (pil_cutout_margin.height - pil_cutout.height) // 2 + pil_cutout.height,
    )
    pil_cutout_margin.paste(pil_cutout, bbox, pil_cutout)
    pil_cutout_margin = pil_cutout_margin.convert("RGB")

    # upload the cutout with margin to the API
    cutout_data = image_to_bytes(pil_cutout_margin)
    stateid_cutout_margin = await ctx.call_async.upload_image(cutout_data)
    app.logger.debug(f"{stateid_cutout_margin=}")

    return stateid_cutout_margin, pil_cutout_margin


async def cutout(ctx: EditorAPIContext, request: Request) -> Response:
    # get information on the caller
    infos = await ctx.call_async.me()
    app.logger.debug(f"{infos['uid']} - {infos['credits']} - calling /cutout")

    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = CutoutParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # validate image input
    if input_data.stateids_input:
        len_stateids_input = len(input_data.stateids_input)
    elif input_data.openaiFileIdRefs:
        len_stateids_input = len(input_data.openaiFileIdRefs)
    else:
        raise ValueError("[cutout] input error: stateids_input or openaiFileIdRefs is required")

    # validate prompt input
    if input_data.prompts is None:
        raise ValueError("[cutout] input error: prompts is required")
    if any(not prompt for prompt in input_data.prompts):
        raise ValueError("[cutout] input error: all the prompts must be not empty")
    if len(input_data.prompts) != len_stateids_input:
        raise ValueError("[cutout] input error: stateids_input and prompts must have the same length")

    # validate background_colors input
    if input_data.background_colors is None:
        input_data.background_colors = ["#ffffff"] * len_stateids_input
    if len(input_data.background_colors) != len_stateids_input:
        raise ValueError("[cutout] input error: stateids_input and background_colors must have the same length")

    # get stateids_input, or create them from openaiFileIdRefs
    if input_data.stateids_input:
        stateids_input = input_data.stateids_input
    elif input_data.openaiFileIdRefs:
        stateids_input = [await ref.get_stateid(ctx) for ref in input_data.openaiFileIdRefs]
    else:
        raise ValueError("[cutout] input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # process the inputs
    async with asyncio.TaskGroup() as tg:
        responses_cutout = [
            tg.create_task(
                process(
                    ctx=ctx,
                    stateid_input=stateid_input,
                    background_color=background_color,
                    prompt=prompt,
                )
            )
            for stateid_input, prompt, background_color in zip(
                stateids_input,
                input_data.prompts,
                input_data.background_colors,
                strict=True,
            )
        ]
    results_cutout = [r.result() for r in responses_cutout]
    stateids_cutout = [r[0] for r in results_cutout]
    pils_cutout = [r[1] for r in results_cutout]

    # get infos on the caller
    infos = await ctx.call_async.me()
    app.logger.debug(f"{infos['uid']} - {infos['credits']} - done /cutout")

    # build output response
    output_data = CutoutOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(
                image=cutout,
                name=f"cutout_{i}",
            )
            for i, cutout in enumerate(pils_cutout)
        ],
        stateids_undo=stateids_input,
        stateids_output=stateids_cutout,
        credits_left=infos["credits"],
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
