from finegrain import EditorAPIContext, ImageOutParams, OKResult, OKResultWithImage


async def test_erase(
    fgctx: EditorAPIContext,
    coffee_plant_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(coffee_plant_bytes)

    bbox_r = await fgctx.call_async.infer_bbox(st_input, product_name="coffee cup")
    assert isinstance(bbox_r, OKResult)

    segment_r = await fgctx.call_async.segment(st_input, bbox=bbox_r.bbox)
    assert isinstance(segment_r, OKResult)

    erase_r = await fgctx.call_async.erase(
        image_state_id=st_input,
        mask_state_id=segment_r.state_id,
        mode="express",
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
    )
    assert isinstance(erase_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-erase-coffee-plant-cup-erased.webp", "wb") as f:
            f.write(erase_r.image)
