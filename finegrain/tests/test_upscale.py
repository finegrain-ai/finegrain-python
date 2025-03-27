from finegrain import EditorAPIContext, ImageOutParams, OKResultWithImage


async def test_upscale(
    fgctx: EditorAPIContext,
    sofa_cushion_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(sofa_cushion_bytes)

    upscaled_r = await fgctx.call_async.upscale(
        st_input,
        with_image=ImageOutParams(resolution="FULL", image_format="WEBP"),
    )
    assert isinstance(upscaled_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-upscale-sofa-cushion.webp", "wb") as f:
            f.write(upscaled_r.image)
