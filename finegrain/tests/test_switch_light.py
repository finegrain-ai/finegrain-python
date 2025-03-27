from finegrain import EditorAPIContext, ImageOutParams, OKResult, OKResultWithImage


async def test_switch_light(
    fgctx: EditorAPIContext,
    table_lamp_bytes: bytes,
    output_dir: str | None,
) -> None:
    st_input = await fgctx.call_async.upload_image(table_lamp_bytes)

    switch_r = await fgctx.call_async.switch_light(st_input)
    assert isinstance(switch_r, OKResult)

    paramd_r = await fgctx.call_async.set_light_params(
        switch_r.state_id,
        brightness=0.5,
        warmth=1.5,
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
    )
    assert isinstance(paramd_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-switch-light-table-lamp.webp", "wb") as f:
            f.write(paramd_r.image)
