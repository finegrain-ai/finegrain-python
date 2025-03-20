from typing import get_args

from finegrain import ErrorResult, Mode, StateID
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
) -> StateID:
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
        raise ValueError(f"[Erase] internal erase error: {result_erase.error}")
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
        raise ValueError("[Erase] input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # validate the inputs
    if input_data.prompts is None:
        raise ValueError("[Erase] input error: prompts is required")
    if any(not prompt for prompt in input_data.prompts):
        raise ValueError("[Erase] input error: all the prompts must be not empty")
    if len(stateids_input) != len(input_data.prompts):
        raise ValueError("[Erase] input error: stateids_input and prompts must have the same length")
    if input_data.mode not in get_args(Mode):
        raise ValueError("[Erase] input error: invalid mode")

    # process the inputs
    stateids_erased = [
        await process(ctx, stateid_input, prompt, input_data.mode)
        for stateid_input, prompt in zip(stateids_input, input_data.prompts, strict=True)
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
