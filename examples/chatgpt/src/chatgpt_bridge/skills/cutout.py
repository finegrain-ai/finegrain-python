import io

from PIL import Image
from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContextCached
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse, StateID, json_error


class CutoutParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    background_colors: list[str] | None = None
    object_names: list[str] | None = None


class CutoutOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_undo: list[StateID]
    stateids_output: list[StateID]


async def process(
    ctx: EditorAPIContextCached,
    stateid_input: StateID,
    background_color: str,
    object_name: str,
) -> tuple[Image.Image, StateID]:
    # queue skills/infer-bbox
    stateid_bbox = await ctx.skill_bbox(stateid_image=stateid_input, product_name=object_name)
    app.logger.debug(f"{stateid_bbox=}")

    # queue skills/segment
    stateid_mask = await ctx.skill_segment(stateid_bbox=stateid_bbox)
    app.logger.debug(f"{stateid_mask=}")

    # queue skills/cutout
    stateid_cutout = await ctx.skill_cutout(stateid_image=stateid_input, stateid_mask=stateid_mask)
    app.logger.debug(f"{stateid_cutout=}")

    cutout = await ctx.download_image(stateid_image=stateid_cutout)

    # paste cutout onto a blank image, with margins
    cutout_margin = Image.new(
        mode="RGBA",
        size=(int(1.618 * cutout.width), int(1.618 * cutout.height)),
        color=background_color,
    )
    bbox = (
        (cutout_margin.width - cutout.width) // 2,
        (cutout_margin.height - cutout.height) // 2,
        (cutout_margin.width - cutout.width) // 2 + cutout.width,
        (cutout_margin.height - cutout.height) // 2 + cutout.height,
    )
    cutout_margin.paste(cutout, bbox, cutout)
    cutout_margin = cutout_margin.convert("RGB")

    # upload the cutout with margin to the API
    cutout_margin_data = io.BytesIO()
    cutout_margin.save(cutout_margin_data, format="JPEG")
    stateid_cutout_margin = await ctx.create_state(file=cutout_margin_data)
    app.logger.debug(f"{stateid_cutout_margin=}")

    return cutout_margin, stateid_cutout_margin


async def _cutout(ctx: EditorAPIContextCached, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = CutoutParams(**input_json)
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

    # validate input data
    if input_data.object_names is None:
        return json_error("object_names is required", 400)
    if len(stateids_input) != len(input_data.object_names):
        return json_error("stateids_input and object_names must have the same length", 400)
    for object_name in input_data.object_names:
        if not object_name:
            return json_error("object name cannot be empty", 400)
    if input_data.background_colors is None:
        input_data.background_colors = ["#ffffff"] * len(stateids_input)
    if len(input_data.background_colors) != len(stateids_input):
        return json_error("stateids_input and background_colors must have the same length", 400)

    # process the inputs
    outputs = [
        await process(
            ctx=ctx,
            object_name=object_name,
            stateid_input=stateid_input,
            background_color=background_color,
        )
        for stateid_input, object_name, background_color in zip(
            stateids_input,
            input_data.object_names,
            input_data.background_colors,
            strict=True,
        )
    ]
    cutouts = [output[0] for output in outputs]
    stateids_output = [output[1] for output in outputs]

    # build output response
    output_data = CutoutOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(
                image=cutout,
                name=f"cutout_{i}",
            )
            for i, cutout in enumerate(cutouts)
        ],
        stateids_undo=stateids_input,
        stateids_output=stateids_output,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
