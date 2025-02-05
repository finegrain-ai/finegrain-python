from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContextCached
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse, StateID, json_error


class ShadowParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    background_colors: list[str] | None = None
    object_names: list[str] | None = None


class ShadowOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContextCached,
    stateid_input: StateID,
    object_name: str,
    background_color: str,
) -> StateID:
    # queue skills/infer-bbox
    stateid_bbox = await ctx.skill_bbox(stateid_image=stateid_input, product_name=object_name)
    app.logger.debug(f"{stateid_bbox=}")

    # queue skills/segment
    stateid_mask = await ctx.skill_segment(stateid_bbox=stateid_bbox)
    app.logger.debug(f"{stateid_mask=}")

    # queue skills/cutout
    stateid_cutout = await ctx.skill_cutout(stateid_image=stateid_input, stateid_mask=stateid_mask)
    app.logger.debug(f"{stateid_cutout=}")

    # queue skills/shadow
    stateid_shadow = await ctx.skill_shadow(stateid_cutout=stateid_cutout, background_color="transparent")
    app.logger.debug(f"{stateid_shadow=}")

    # queue skills/set-background-color
    stateid_shadow_colorbg = await ctx.skill_set_bgcolor(stateid_image=stateid_shadow, color=background_color)
    app.logger.debug(f"{stateid_shadow_colorbg=}")

    return stateid_shadow_colorbg


async def _shadow(ctx: EditorAPIContextCached, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = ShadowParams(**input_json)
    app.logger.debug(f"{input_data=}")

    # get stateids_input, or create them from openaiFileIdRefs
    if input_data.stateids_input:
        stateids_input = input_data.stateids_input
    elif input_data.openaiFileIdRefs:
        stateids_input: list[str] = []
        for oai_ref in input_data.openaiFileIdRefs:
            if oai_ref.download_link:
                stateid_input = await ctx.create_state(oai_ref.download_link)
                stateids_input.append(stateid_input)
    else:
        return json_error("stateids_input or openaiFileIdRefs is required", 400)
    app.logger.debug(f"{stateids_input=}")

    # validate object_names
    if input_data.object_names is None:
        return json_error("object_names is required", 400)
    if len(stateids_input) != len(input_data.object_names):
        return json_error("stateids_input and object_names must have the same length", 400)
    for object_name in input_data.object_names:
        if not object_name:
            return json_error("object name cannot be empty", 400)

    # validate background_colors
    if input_data.background_colors is None:
        input_data.background_colors = ["#ffffff"] * len(stateids_input)
    if len(stateids_input) != len(input_data.background_colors):
        return json_error("stateids_input and background_colors must have the same length", 400)
    for background_color in input_data.background_colors:
        if not background_color:
            return json_error("background color cannot be empty", 400)

    # process the inputs
    stateids_shadow = [
        await process(
            ctx=ctx,
            stateid_input=stateid_input,
            object_name=object_name,
            background_color=background_color,
        )
        for stateid_input, object_name, background_color in zip(
            stateids_input,
            input_data.object_names,
            input_data.background_colors,
            strict=True,
        )
    ]

    # download output images
    shadow_imgs = [
        await ctx.download_image(stateid_image=stateid_shadow)  #
        for stateid_shadow in stateids_shadow
    ]

    # build output response
    output_data = ShadowOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(
                image=shadow_img,
                name=f"shadow_{i}",
            )
            for i, shadow_img in enumerate(shadow_imgs)
        ],
        stateids_output=stateids_shadow,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
