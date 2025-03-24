import asyncio
from typing import get_args

from finegrain import ErrorResult, Mode
from quart import Request, Response
from quart import current_app as app
from quart.datastructures import FileStorage

from figma_bridge.utils import get_ctx


async def erase(request: Request) -> Response:
    # parse input data
    input_form = await request.form
    input_files = await request.files
    api_key = input_form.get("api_key")
    api_mode = input_form.get("api_mode")
    image_format = input_form.get("image_format")
    prompt = input_form.get("prompt")
    image = input_files.get("image")

    # validate api_key input
    assert api_key, "Missing 'api_key' field"
    assert isinstance(api_key, str), "Invalid 'api_key' field"

    # validate mode input
    assert api_mode, "Missing 'api_mode' field"
    assert api_mode in get_args(Mode), f"Invalid 'api_mode' selection, got {api_mode}"

    # validate prompt input
    assert prompt, "Missing 'prompt' field"
    assert isinstance(prompt, str), "Invalid 'prompt' field"

    # validate image_format input
    assert image_format, "Missing 'image_format' field"
    assert image_format in ["JPEG", "PNG", "WEBP"], f"Invalid 'image_format' selection, got {image_format}"

    # validate image input
    assert image, "Missing 'image' file"
    assert isinstance(image, FileStorage), "Invalid 'image' file"

    async with get_ctx(api_key) as ctx:
        # upload the image to the API
        stateid_input = await ctx.call_async.upload_image(file=image.stream)
        app.logger.debug(f"{stateid_input=}")

        # call detect
        result_detect = await ctx.call_async.detect(state_id=stateid_input, prompt=prompt)
        assert not isinstance(result_detect, ErrorResult), f"Detection failed: {result_detect.error}"
        detections = result_detect.results
        assert len(detections) > 0, f"No detections found for prompt: {prompt}"
        detections = detections[:15]  # cap to 15 detections
        app.logger.debug(f"{detections=}")

        # call segment on each detection
        async with asyncio.TaskGroup() as tg:
            responses_segment = [
                tg.create_task(
                    ctx.call_async.segment(
                        state_id=stateid_input,
                        bbox=detection.bbox,
                    )
                )
                for detection in detections
            ]
        results_segment = [r.result() for r in responses_segment]
        for r in results_segment:
            assert not isinstance(r, ErrorResult), f"Segmentation failed: {r.error}"
        stateids_segment = [r.state_id for r in results_segment]
        app.logger.debug(f"{stateids_segment=}")

        # call merge-masks
        if len(stateids_segment) == 1:
            stateid_merge = stateids_segment[0]
        else:
            result_merge = await ctx.call_async.merge_masks(
                state_ids=stateids_segment,
                operation="union",
            )
            assert not isinstance(result_merge, ErrorResult), f"Merging segmentations failed: {result_merge.error}"
            stateid_merge = result_merge.state_id
        app.logger.debug(f"{stateid_merge=}")

        # call erase skill
        result_erase = await ctx.call_async.erase(
            image_state_id=stateid_input,
            mask_state_id=stateid_merge,
            mode=api_mode,
        )
        assert not isinstance(result_erase, ErrorResult), f"Eraser failed: {result_erase.error}"
        stateid_erase = result_erase.state_id
        app.logger.debug(f"{stateid_erase=}")

        # download images from API
        data_erase = await ctx.get_image(
            state_id=stateid_erase,
            image_format=image_format,
            resolution="FULL",
        )

    # build output response
    output_response = Response(response=data_erase, content_type="image/jpeg")
    return output_response
