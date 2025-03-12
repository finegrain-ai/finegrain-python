import argparse
import asyncio
from typing import Any

from typing_extensions import TypeIs

from finegrain import EditorAPIContext, EraseResultWithImage, ErrorResult


def is_error(result: Any) -> TypeIs[ErrorResult]:
    if isinstance(result, ErrorResult):
        raise RuntimeError(result.error)
    return False


async def co(ctx: EditorAPIContext, prompt: str, path_in: str, path_out: str) -> None:
    await ctx.login()
    await ctx.sse_start()

    with open(path_in, "rb") as f:
        st_input = await ctx.call_async.upload_image(f)

    bbox_r = await ctx.call_async.infer_bbox(st_input, prompt)
    assert not is_error(bbox_r)
    print(f"Got bounding box: {bbox_r.bbox}")

    mask_r = await ctx.call_async.segment(bbox_r.state_id)
    assert not is_error(mask_r)

    erased_r = await ctx.call_async.erase(st_input, mask_r.state_id, mode="express", with_image=True)
    assert not is_error(erased_r)
    assert isinstance(erased_r, EraseResultWithImage)

    with open(path_out, "wb") as f:
        f.write(erased_r.image)
    print(f"Output image in {path_out}")

    await ctx.sse_stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--credentials",
        type=str,
        required=True,
        help="API credentials (FGAPI-... or user@example.com:P455w0rD)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="What to remove?")
    parser.add_argument("--input-file", type=str, default="before.webp", help="Input file path")
    parser.add_argument("--output-file", type=str, default="after.webp", help="Output file path")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL")
    args = parser.parse_args()

    ctx = EditorAPIContext(
        credentials=args.credentials,
        base_url=args.base_url,
        user_agent="finegrain-python-example",
    )
    asyncio.run(co(ctx, args.prompt, args.input_file, args.output_file))
