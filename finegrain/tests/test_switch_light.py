from finegrain import EditorAPIContext, ImageOutParams, OKResultWithImage


async def test_switch_light(
    fgctx: EditorAPIContext,
    table_lamp_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(table_lamp_bytes)

    switch_r = await fgctx.call_async.switch_light(
        st_input,
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
    )
    assert isinstance(switch_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-switch-light-table-lamp-base.webp", "wb") as f:
            f.write(switch_r.image)

    paramd_r = await fgctx.call_async.set_light_params(
        switch_r.state_id,
        brightness=0.8,
        warmth=1.5,
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
    )
    assert isinstance(paramd_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-switch-light-table-lamp-tweaked.webp", "wb") as f:
            f.write(paramd_r.image)
