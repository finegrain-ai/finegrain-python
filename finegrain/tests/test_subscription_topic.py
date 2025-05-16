import pytest

from finegrain import EditorAPIContext, ImageOutParams, OKResult, OKResultWithImage


@pytest.mark.parametrize("subscription_topic", ["fg-test-topic", None])
async def test_subscription_topic(
    fx_base_url: str,
    fx_credentials: str,
    fx_verify: bool,
    sofa_cushion_bytes: bytes,
    subscription_topic: str | None,
    output_dir: str | None,
) -> None:
    ctx = EditorAPIContext(
        base_url=fx_base_url,
        credentials=fx_credentials,
        verify=fx_verify,
        user_agent="finegrain-python-tests",
        subscription_topic=subscription_topic,
    )

    await ctx.login()
    await ctx.sse_start()

    create_r = await ctx.call_async.create_state(
        file=sofa_cushion_bytes,
        meta={"test-key": "test-value"},
    )
    assert isinstance(create_r, OKResult)
    assert create_r.meta["test-key"] == "test-value"

    # to test credits update mechanism
    assert isinstance(ctx.credits, int)
    ctx.credits = None

    infer_ms_r = await ctx.call_async.infer_product_name(create_r.state_id)
    assert isinstance(infer_ms_r, OKResult)
    assert infer_ms_r.meta["product_name"] == "sofa"

    if subscription_topic is not None:
        assert ctx.credits is None
        await ctx.me()
    assert isinstance(ctx.credits, int)

    # test segment (multi-step)

    segment_r = await ctx.call_async.segment(
        create_r.state_id,
        prompt="sofa",
        with_image=ImageOutParams(resolution="DISPLAY", image_format="WEBP"),
        mask_quality="low",
    )
    assert isinstance(segment_r, OKResultWithImage)

    if output_dir:
        with open(f"{output_dir}/test-subscription-topic-segment-sofa.webp", "wb") as f:
            f.write(segment_r.image)

    await ctx.sse_stop()
