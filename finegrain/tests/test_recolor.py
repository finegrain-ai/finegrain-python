import asyncio

from finegrain import EditorAPIContext, ImageOutParams, OKResult, OKResultWithImage


async def test_recolor(
    fgctx: EditorAPIContext,
    sofa_cushion_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(sofa_cushion_bytes)

    bboxes = [
        (109, 264, 1300, 883),
        (847, 289, 1182, 499),
        (153, 748, 231, 853),
        (1210, 629, 1298, 733),
    ]

    segment_results = await asyncio.gather(*[fgctx.call_async.segment(st_input, bbox) for bbox in bboxes])
    for r in segment_results:
        assert isinstance(r, OKResult)
    segment_state_ids = [r.state_id for r in segment_results]

    merged_r = await fgctx.call_async.merge_masks(segment_state_ids, "difference")
    assert isinstance(merged_r, OKResult)

    recolored_r = await fgctx.call_async.recolor(
        st_input,
        merged_r.state_id,
        color="#5a4848",
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
    )
    assert isinstance(recolored_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-recolor-sofa-cushion.webp", "wb") as f:
            f.write(recolored_r.image)
