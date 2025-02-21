# Finegrain API - Python client

This is a client for the [Finegrain](https://finegrain.ai) API. It requires Python 3.12+ and is designed for asynchronous code using asyncio. It depends on httpx and [httpx_sse](https://github.com/florimondmanca/httpx-sse).

## Usage

Here is an example script to erase an object from an image by prompt.

```py
import argparse
import asyncio

from finegrain import EditorAPIContext, EraseResultWithImage, ErrorResult


async def co(ctx: EditorAPIContext, prompt: str, path_in: str, path_out: str) -> None:
    await ctx.login()
    await ctx.sse_start()

    with open(path_in, "rb") as f:
        st_input = await ctx.call_async.upload_image(f)

    bbox_r = await ctx.call_async.infer_bbox(st_input, prompt)
    assert not isinstance(bbox_r, ErrorResult)
    print(f"Got bounding box: {bbox_r.bbox}")

    mask_r = await ctx.call_async.segment(bbox_r.state_id)
    assert not isinstance(mask_r, ErrorResult)

    erased_r = await ctx.call_async.erase(st_input, mask_r.state_id, mode="express", with_image=True)
    assert isinstance(erased_r, EraseResultWithImage)

    with open(path_out, "wb") as f:
        f.write(erased_r.image)
    print(f"Output image in {path_out}")

    await ctx.sse_stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, required=True, help="API login")
    parser.add_argument("--password", type=str, required=True, help="API password")
    parser.add_argument("--prompt", type=str, required=True, help="What to remove?")
    parser.add_argument("--input-file", type=str, default="before.webp", help="Input file path")
    parser.add_argument("--output-file", type=str, default="after.webp", help="Output file path")
    args = parser.parse_args()

    ctx = EditorAPIContext(user=args.user, password=args.password)
    asyncio.run(co(ctx, args.prompt, args.input_file, args.output_file))
```
