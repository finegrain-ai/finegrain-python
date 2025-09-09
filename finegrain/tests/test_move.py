import asyncio

from finegrain import EditorAPIContext, ImageOutParams, OKResult, OKResultWithImage


async def test_move(
    fgctx: EditorAPIContext,
    coffee_plant_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(coffee_plant_bytes)

    bbox_r = await fgctx.call_async.infer_bbox(st_input, product_name="coffee cup")
    assert isinstance(bbox_r, OKResult)

    segment_r = await fgctx.call_async.segment(st_input, bbox=bbox_r.bbox)
    assert isinstance(segment_r, OKResult)

    erase_r, cutout_r = await asyncio.gather(
        fgctx.call_async.erase(st_input, segment_r.state_id),
        fgctx.call_async.cutout(st_input, segment_r.state_id),
    )
    assert isinstance(erase_r, OKResult)
    assert isinstance(cutout_r, OKResult)

    blend_r = await fgctx.call_async.blend(
        image_state_id=erase_r.state_id,
        cutout_state_id=cutout_r.state_id,
        bbox=(871, 1445, 1106, 1684),
        flip=True,
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
    )
    assert isinstance(blend_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-move-coffee-plant.webp", "wb") as f:
            f.write(blend_r.image)
