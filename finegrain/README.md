# Finegrain API - Python client

This is a client for the [Finegrain](https://finegrain.ai) API. It requires Python 3.12+ and is designed for asynchronous code using asyncio. It depends on httpx and [httpx_sse](https://github.com/florimondmanca/httpx-sse).

## Usage

Here is an example script to erase an object from an image by prompt.

```py
import argparse
import asyncio

from finegrain import EditorAPIContext


async def co(ctx: EditorAPIContext, prompt: str, path_in: str, path_out: str) -> None:
    await ctx.login()
    await ctx.sse_start()

    with open(path_in, "rb") as f:
        response = await ctx.request("POST", "state/upload", files={"file": f})
    st_input = response.json()["state"]

    st_boxed = await ctx.ensure_skill(f"infer-bbox/{st_input}", {"product_name": prompt})
    st_mask = await ctx.ensure_skill(f"segment/{st_boxed}")
    st_erased = await ctx.ensure_skill(f"erase/{st_input}/{st_mask}", {"mode": "free"})

    response = await ctx.request(
        "GET",
        f"state/image/{st_erased}",
        params={"format": "WEBP", "resolution": "DISPLAY"},
    )

    with open(path_out, "wb") as f:
        f.write(response.content)

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
