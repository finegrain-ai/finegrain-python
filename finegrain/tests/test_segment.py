from finegrain import EditorAPIContext, ImageOutParams, OKResult, OKResultWithImage


async def test_segment_bbox(
    fgctx: EditorAPIContext,
    coffee_plant_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(coffee_plant_bytes)

    bbox_r = await fgctx.call_async.infer_bbox(st_input, product_name="coffee cup")
    assert isinstance(bbox_r, OKResult)

    segment_r = await fgctx.call_async.segment(
        st_input,
        bbox=bbox_r.bbox,
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
    )
    assert isinstance(segment_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-segment-coffee-plant-bbox.webp", "wb") as f:
            f.write(segment_r.image)


async def test_segment_prompt_param(
    fgctx: EditorAPIContext,
    coffee_plant_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(coffee_plant_bytes)

    segment_r = await fgctx.call_async.segment(
        st_input,
        prompt="coffee cup",
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
        mask_quality="low",
    )
    assert isinstance(segment_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-segment-coffee-plant-prompt-param.webp", "wb") as f:
            f.write(segment_r.image)


async def test_segment_prompt_meta(
    fgctx: EditorAPIContext,
    coffee_plant_bytes: bytes,
    output_dir: str | None,
) -> None:
    create_r = await fgctx.call_async.create_state(
        file=coffee_plant_bytes,
        meta={"prompt": "coffee cup"},
    )
    assert isinstance(create_r, OKResult)

    segment_r = await fgctx.call_async.segment(
        create_r.state_id,
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
        mask_quality="low",
    )
    assert isinstance(segment_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-segment-coffee-plant-prompt-meta.webp", "wb") as f:
            f.write(segment_r.image)
