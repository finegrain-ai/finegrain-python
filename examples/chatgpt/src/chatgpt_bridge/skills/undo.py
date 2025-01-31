from finegrain import EditorAPIContext
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.utils import OpenaiFileResponse, StateID, download_image, json_error


class UndoParams(BaseModel):
    stateids_undo: list[StateID] | None = None


class UndoOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]


async def _undo(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"json payload: {input_json}")
    input_data = UndoParams(**input_json)
    app.logger.debug(f"parsed payload: {input_data}")

    # validate input data
    if not input_data.stateids_undo:
        return json_error("stateids_undo is required", 400)

    # download the image
    images = [
        await download_image(ctx, stateid=stateid)  #
        for stateid in input_data.stateids_undo
    ]

    # build output response
    output_data = UndoOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(
                image=image,
                name=f"undo_{i}",
            )
            for i, image in enumerate(images)
        ],
        stateids_output=input_data.stateids_undo,
    )
    app.logger.debug(f"output payload: {output_data}")
    output_response = jsonify(output_data.model_dump())
    return output_response
