from finegrain import EditorAPIContext
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse, StateID, create_state, download_image, json_error


class UpscaleParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None


class UpscaleOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def _upscale(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"json payload: {input_json}")
    input_data = UpscaleParams(**input_json)
    app.logger.debug(f"parsed payload: {input_data}")

    # get stateids_input, or create them from openaiFileIdRefs
    if input_data.stateids_input:
        stateids_input = input_data.stateids_input
    elif input_data.openaiFileIdRefs:
        stateids_input: list[StateID] = []
        for oai_ref in input_data.openaiFileIdRefs:
            if oai_ref.download_link:
                stateid_input = await create_state(ctx, oai_ref.download_link)
                stateids_input.append(stateid_input)
    else:
        return json_error("stateids_input or openaiFileIdRefs is required", 400)
    app.logger.debug(f"stateids_input: {stateids_input}")

    # queue skills/upscale
    stateids_upscaled = [
        await ctx.ensure_skill(url=f"upscale/{stateid_input}")  #
        for stateid_input in stateids_input
    ]
    app.logger.debug(f"stateids_upscaled: {stateids_upscaled}")

    # download output images
    upscaled_images = [
        await download_image(ctx, stateid_upscaled)  #
        for stateid_upscaled in stateids_upscaled
    ]

    # build output response
    output_data = UpscaleOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(image=upscaled_img, name=f"upscaled_{i}")
            for i, upscaled_img in enumerate(upscaled_images)
        ],
        stateids_output=stateids_upscaled,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"output payload: {output_data}")
    output_response = jsonify(output_data.model_dump())
    return output_response
