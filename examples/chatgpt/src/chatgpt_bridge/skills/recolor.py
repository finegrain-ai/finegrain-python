from pydantic import BaseModel
from quart import Request, Response, jsonify
from quart import current_app as app

from chatgpt_bridge.context import EditorAPIContextCached
from chatgpt_bridge.utils import OpenaiFileIdRef, OpenaiFileResponse, StateID, json_error


class RecolorParams(BaseModel):
    openaiFileIdRefs: list[OpenaiFileIdRef] | None = None  # noqa: N815
    stateids_input: list[StateID] | None = None
    positive_object_names: list[list[str]] | None = None
    negative_object_names: list[list[str]] | None = None
    object_colors: list[str] | None = None


class RecolorOutput(BaseModel):
    openaiFileResponse: list[OpenaiFileResponse]  # noqa: N815
    stateids_output: list[StateID]
    stateids_undo: list[StateID]


async def process(
    ctx: EditorAPIContextCached,
    object_color: str,
    stateid_input: str,
    positive_object_names: list[str],
    negative_object_names: list[str],
) -> str:
    # queue skills/infer-bbox for positive objects
    stateids_bbox_positive = []
    for name in positive_object_names:
        stateid_bbox_positive = await ctx.skill_bbox(stateid_image=stateid_input, product_name=name)
        app.logger.debug(f"{stateid_bbox_positive=}")
        stateids_bbox_positive.append(stateid_bbox_positive)
    app.logger.debug(f"{stateids_bbox_positive=}")

    # queue skills/infer-bbox for negative objects
    stateids_bbox_negative = []
    for name in negative_object_names:
        stateid_bbox_negative = await ctx.skill_bbox(stateid_image=stateid_input, product_name=name)
        app.logger.debug(f"{stateid_bbox_negative=}")
        stateids_bbox_negative.append(stateid_bbox_negative)
    app.logger.debug(f"{stateids_bbox_negative=}")

    # queue skills/segment for positive objects
    stateids_mask_positive = []
    for stateid_bbox in stateids_bbox_positive:
        stateid_mask_positive = await ctx.skill_segment(stateid_bbox=stateid_bbox)
        app.logger.debug(f"{stateid_mask_positive=}")
        stateids_mask_positive.append(stateid_mask_positive)
    app.logger.debug(f"{stateids_mask_positive=}")

    # queue skills/segment for negative objects
    stateids_mask_negative = []
    for stateid_bbox in stateids_bbox_negative:
        stateid_mask_negative = await ctx.skill_segment(stateid_bbox=stateid_bbox)
        app.logger.debug(f"{stateid_mask_negative=}")
        stateids_mask_negative.append(stateid_mask_negative)
    app.logger.debug(f"{stateids_mask_negative=}")

    # queue skills/merge-masks for positive objects
    if len(stateids_mask_positive) == 1:
        stateid_mask_positive_union = stateids_mask_positive[0]
    else:
        stateid_mask_positive_union = await ctx.skill_merge_masks(
            stateids=tuple(stateids_mask_positive),
            operation="union",
        )
    app.logger.debug(f"{stateid_mask_positive_union=}")

    # queue skills/merge-masks for negative objects
    if len(stateids_mask_negative) == 0:
        stateid_mask_negative_union = None
    elif len(stateids_mask_negative) == 1:
        stateid_mask_negative_union = stateids_mask_negative[0]
    else:
        stateid_mask_negative_union = await ctx.skill_merge_masks(
            stateids=tuple(stateids_mask_negative),
            operation="union",
        )
    app.logger.debug(f"{stateid_mask_negative_union=}")

    # queue skills/merge-masks for difference between positive and negative masks
    if stateid_mask_negative_union is not None:
        stateid_mask_difference = await ctx.skill_merge_masks(
            stateids=(stateid_mask_positive_union, stateid_mask_negative_union),
            operation="difference",
        )
    else:
        stateid_mask_difference = stateid_mask_positive_union
    app.logger.debug(f"{stateid_mask_difference=}")

    # queue skills/recolor
    stateid_recolor = await ctx.skill_recolor(
        stateid_image=stateid_input,
        stateid_mask=stateid_mask_difference,
        color=object_color,
    )
    app.logger.debug(f"{stateid_recolor=}")

    return stateid_recolor


async def _recolor(ctx: EditorAPIContextCached, request: Request) -> Response:
    # parse input data
    input_json = await request.get_json()
    app.logger.debug(f"{input_json=}")
    input_data = RecolorParams(**input_json)
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

    # validate object_colors
    if input_data.object_colors is None:
        return json_error("object_colors is required", 400)
    if len(stateids_input) != len(input_data.object_colors):
        return json_error("stateids_input and object_colors must have the same length", 400)

    # validate positive_object_names
    if input_data.positive_object_names is None:
        return json_error("positive_object_names is required", 400)
    if len(stateids_input) != len(input_data.positive_object_names):
        return json_error("stateids_input and positive_object_names must have the same length", 400)
    for object_names in input_data.positive_object_names:
        if not object_names:
            return json_error("positive object list cannot be empty", 400)
        for object_name in object_names:
            if not object_name:
                return json_error("positive object name cannot be empty", 400)

    # validate negative_object_names
    if input_data.negative_object_names is None:
        input_data.negative_object_names = [[]] * len(stateids_input)
    if len(stateids_input) != len(input_data.negative_object_names):
        return json_error("stateids_input and negative_object_names must have the same length", 400)
    for object_names in input_data.negative_object_names:
        for object_name in object_names:
            if not object_name:
                return json_error("negative object name cannot be empty", 400)

    # process the inputs
    stateids_recolor = [
        await process(
            ctx=ctx,
            object_color=object_color,
            stateid_input=stateid_input,
            positive_object_names=positive_object_names,
            negative_object_names=negative_object_names,
        )
        for stateid_input, positive_object_names, negative_object_names, object_color in zip(
            stateids_input,
            input_data.positive_object_names,
            input_data.negative_object_names,
            input_data.object_colors,
            strict=True,
        )
    ]
    app.logger.debug(f"{stateids_recolor=}")

    # download the output images
    recolor_imgs = [
        await ctx.download_image(stateid_image=stateid_recolor)  #
        for stateid_recolor in stateids_recolor
    ]

    # build output response
    output_data = RecolorOutput(
        openaiFileResponse=[
            OpenaiFileResponse.from_image(
                image=recolor_img,
                name=f"recolored_{i}",
            )
            for i, recolor_img in enumerate(recolor_imgs)
        ],
        stateids_output=stateids_recolor,
        stateids_undo=stateids_input,
    )
    app.logger.debug(f"{output_data=}")
    output_response = jsonify(output_data.model_dump())
    return output_response
