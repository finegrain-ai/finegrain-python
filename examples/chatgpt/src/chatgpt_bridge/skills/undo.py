from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContextCached
from chatgpt_bridge.utils import OpenaiFileResponse, StateID, json_error


class UndoParams(BaseModel):
    stateids_undo: list[StateID] | None = None


class UndoOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]


async def _undo(ctx: EditorAPIContextCached, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = UndoParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # validate input data
    if not input_data.stateids_undo:
        return json_error("stateids_undo is required", 400)

    # download the image
    images = [
        await ctx.download_image(stateid_image=stateid)  #
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
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
