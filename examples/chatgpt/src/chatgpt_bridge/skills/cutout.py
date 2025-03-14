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
    object_names: list[str] | None = None


class CutoutOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_undo: list[StateID]
    stateids_output: list[StateID]


async def process(
    ctx: EditorAPIContext,
    stateid_input: StateID,
    background_color: str,
    object_name: str,
) -> tuple[Image.Image, StateID]:
    # call infer-bbox
    result_bbox = await ctx.call_async.infer_bbox(
        state_id=stateid_input,
        product_name=object_name,
    )
    if isinstance(result_bbox, ErrorResult):
        raise ValueError(f"Cutout internal infer_bbox error: {result_bbox.error}")
    bbox = result_bbox.bbox
    app.logger.debug(f"{bbox=}")

    # call segment
    result_segment = await ctx.call_async.segment(
        state_id=stateid_input,
        bbox=bbox,
    )
    if isinstance(result_segment, ErrorResult):
        raise ValueError(f"Cutout internal segment error: {result_segment.error}")
    stateid_segment = result_segment.state_id
    app.logger.debug(f"{stateid_segment=}")

    # call cutout
    result_cutout = await ctx.call_async.cutout(
        image_state_id=stateid_input,
        mask_state_id=stateid_segment,
    )
    if isinstance(result_cutout, ErrorResult):
        raise ValueError(f"Cutout internal cutout error: {result_cutout.error}")
    stateid_cutout = result_cutout.state_id
    app.logger.debug(f"{stateid_cutout=}")

    # download cutout
    cutout = await ctx.call_async.download_pil_image(stateid_cutout)

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
    cutout_data = image_to_bytes(cutout_margin)
    stateid_cutout_margin = await ctx.call_async.upload_image(cutout_data)
    app.logger.debug(f"{stateid_cutout_margin=}")

    return cutout_margin, stateid_cutout_margin


async def _cutout(ctx: EditorAPIContext, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = CutoutParams(**input_json)
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
        raise ValueError("Cutout input error: stateids_input or openaiFileIdRefs is required")
    app.logger.debug(f"{stateids_input=}")

    # validate input data
    if input_data.object_names is None:
        raise ValueError("Cutout input error: object_names is required")
    if len(stateids_input) != len(input_data.object_names):
        raise ValueError("Cutout input error: stateids_input and object_names must have the same length")
    for object_name in input_data.object_names:
        if not object_name:
            raise ValueError("Cutout input error: object name cannot be empty")
    if input_data.background_colors is None:
        input_data.background_colors = ["#ffffff"] * len(stateids_input)
    if len(input_data.background_colors) != len(stateids_input):
        raise ValueError("Cutout input error: stateids_input and background_colors must have the same length")

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
