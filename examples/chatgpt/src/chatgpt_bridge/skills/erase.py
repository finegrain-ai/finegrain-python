from finegrain import EditorAPIContext
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse, StateID, create_state, download_image, json_error


class EraseParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    object_names: list[list[str]] | None = None


class EraseOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContext,
    stateid_input: StateID,
    object_names: list[str],
) -> StateID:
    # queue skills/infer-bbox
    stateids_bbox = []
    for name in object_names:
        stateid_bbox = await ctx.ensure_skill(
            url=f"infer-bbox/{stateid_input}",
            params={"product_name": name},
        )
        app.logger.debug(f"{stateid_bbox=}")
        stateids_bbox.append(stateid_bbox)
    app.logger.debug(f"{stateids_bbox=}")

    # queue skills/segment
    stateids_mask = []
    for stateid_bbox in stateids_bbox:
        stateid_mask = await ctx.ensure_skill(url=f"segment/{stateid_bbox}")
        app.logger.debug(f"{stateid_mask=}")
        stateids_mask.append(stateid_mask)
    app.logger.debug(f"{stateids_mask=}")

    # queue skills/merge-masks for positive objects
    if len(stateids_mask) == 1:
        stateid_mask_union = stateids_mask[0]
    else:
        stateid_mask_union = await ctx.ensure_skill(
            url="merge-masks",
            params={
                "operation": "union",
                "states": stateids_mask,
            },
        )
    app.logger.debug(f"{stateid_mask_union=}")

    # queue skills/erase
    stateid_erased = await ctx.ensure_skill(
        url=f"erase/{stateid_input}/{stateid_mask_union}",
        params={"mode": "free"},
    )
    app.logger.debug(f"{stateid_erased=}")

    return stateid_erased


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
                stateid_input = await create_state(ctx, oai_ref.download_link)
                stateids_input.append(stateid_input)
    else:
        return json_error("stateids_input or openaiFileIdRefs is required", 400)
    app.logger.debug(f"{stateids_input=}")

    # validate the inputs
    if input_data.object_names is None:
        return json_error("object_names is required", 400)
    if len(stateids_input) != len(input_data.object_names):
        return json_error("stateids_input and object_names must have the same length", 400)
    for object_names in input_data.object_names:
        if not object_names:
            return json_error("object list cannot be empty", 400)
        for object_name in object_names:
            if not object_name:
                return json_error("object name cannot be empty", 400)

    # process the inputs
    stateids_erased = [
        await process(ctx, stateid_input, object_names)
        for stateid_input, object_names in zip(stateids_input, input_data.object_names, strict=True)
    ]
    app.logger.debug(f"{stateids_erased=}")

    # download images from API
    erased_imgs = [
        await download_image(ctx=ctx, stateid=stateid_erased_img)  #
        for stateid_erased_img in stateids_erased
    ]

    # build output response
    output_data = EraseOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(
                image=erased_img,
                name=f"erased_{i}",
            )
            for i, erased_img in enumerate(erased_imgs)
        ],
        stateids_output=stateids_erased,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
