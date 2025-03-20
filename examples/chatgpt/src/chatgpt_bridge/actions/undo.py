import asyncio

from finegrain import StateID
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContext
from chatgpt_bridge.utils import OpenaiFileResponse


class UndoParams(BaseModel):
    stateids_undo: list[StateID] | None = None


class UndoOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]


async def undo(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = UndoParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # validate input data
    if not input_data.stateids_undo:
        raise ValueError("[undo] input error: stateids_undo is required")

    # download the image
    async with asyncio.TaskGroup() as tg:
        responses_undo = [
            tg.create_task(
                ctx.call_async.download_pil_image(stateid),
            )
            for stateid in input_data.stateids_undo
        ]
    pils_undo = [r.result() for r in responses_undo]

    # build output response
    output_data = UndoOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(image=image, name=f"undo_{i}")  #
            for i, image in enumerate(pils_undo)
        ],
        stateids_output=input_data.stateids_undo,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
