import argparse
import dataclasses as dc
from typing import Any

from typing_extensions import TypeIs

from finegrain import EditorAPIContext, EraseResultWithImage, ErrorResult


def is_error(result: Any) -> TypeIs[ErrorResult]:
    if isinstance(result, ErrorResult):
        raise RuntimeError(result.error)
    return False


@dc.dataclass(kw_only=True)
class Params:
    prompt: str
    path_in: str


async def co(ctx: EditorAPIContext, params: Params) -> bytes:
    with open(params.path_in, "rb") as f:
        st_input = await ctx.call_async.upload_image(f)

    bbox_r = await ctx.call_async.infer_bbox(st_input, params.prompt)
    assert not is_error(bbox_r)
    print(f"Got bounding box: {bbox_r.bbox}")

    mask_r = await ctx.call_async.segment(bbox_r.state_id)
    assert not is_error(mask_r)

    erased_r = await ctx.call_async.erase(st_input, mask_r.state_id, mode="express", with_image=True)
    assert not is_error(erased_r)
    assert isinstance(erased_r, EraseResultWithImage)

    return erased_r.image


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
    image_bytes = ctx.run_one_sync(co, Params(prompt=args.prompt, path_in=args.input_file))

    with open(args.output_file, "wb") as f:
        f.write(image_bytes)
    print(f"Output image in {args.output_file}")
