import asyncio
from typing import get_args

from finegrain import ErrorResult, Mode, StateID
from PIL import Image
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse


class EraseParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    prompts: list[str] | None = None
    mode: Mode = "standard"


class EraseOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContext,
    stateid_input: StateID,
    prompt: str,
    mode: Mode,
) -> tuple[StateID, Image.Image]:
    # call multi_segment
    stateid_segment = await ctx.call_async.multi_segment(
        state_id=stateid_input,
        prompt=prompt,
    )
    app.logger.debug(f"{stateid_segment=}")

    # call erase skill
    result_erase = await ctx.call_async.erase(
        image_state_id=stateid_input,
        mask_state_id=stateid_segment,
        mode=mode,
    )
    if isinstance(result_erase, ErrorResult):
        raise ValueError(f"[erase] internal erase error: {result_erase.error}")
    stateid_erase = result_erase.state_id
    app.logger.debug(f"{stateid_erase=}")

    # download output image
    pil_erase = await ctx.call_async.download_pil_image(stateid_erase)

    return stateid_erase, pil_erase


async def erase(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = EraseParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # validate image input
    if input_data.stateids_input:
        len_stateids_input = len(input_data.stateids_input)
    elif input_data.openaiFileIdRefs:
        len_stateids_input = len(input_data.openaiFileIdRefs)
    else:
        raise ValueError("[erase] input error: stateids_input or openaiFileIdRefs is required")

    # validate prompt input
    if input_data.prompts is None:
        raise ValueError("[erase] input error: prompts is required")
    if any(not prompt for prompt in input_data.prompts):
        raise ValueError("[erase] input error: all the prompts must be not empty")
    if len(input_data.prompts) != len_stateids_input:
        raise ValueError("[erase] input error: stateids_input and prompts must have the same length")

    # validate mode input
    if input_data.mode not in get_args(Mode):
        raise ValueError("[erase] input error: invalid mode")

    # get stateids_input, or create them from openaiFileIdRefs
    if input_data.stateids_input:
        stateids_input = input_data.stateids_input
    elif input_data.openaiFileIdRefs:
        stateids_input = [await ref.get_stateid(ctx) for ref in input_data.openaiFileIdRefs]
    else:
        raise ValueError("[erase] input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # process the inputs
    async with asyncio.TaskGroup() as tg:
        responses_erase = [
            tg.create_task(
                process(
                    ctx=ctx,
                    stateid_input=stateid_input,
                    prompt=prompt,
                    mode=input_data.mode,
                )
            )
            for stateid_input, prompt in zip(stateids_input, input_data.prompts, strict=True)
        ]
    results_erase = [r.result() for r in responses_erase]
    stateids_erase = [r[0] for r in results_erase]
    pils_erase = [r[1] for r in results_erase]

    # build output response
    data_output = EraseOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(image=erased_img, name=f"erased_{i}")
            for i, erased_img in enumerate(pils_erase)
        ],
        stateids_output=stateids_erase,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{data_output=}")
    output_response = jsonify(data_output.model_dump())
    return output_response
