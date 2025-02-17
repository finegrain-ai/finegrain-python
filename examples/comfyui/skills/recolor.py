from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext
from ..utils.image import (
    image_to_bytes,
    image_to_tensor,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    mask: torch.Tensor
    color: str


async def _process(ctx: EditorAPIContext, params: Params) -> torch.Tensor:
    # convert tensors to PIL images
    image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))
    mask_pil = tensor_to_image(params.mask.unsqueeze(0))

    # make some assertions
    assert image_pil.size == mask_pil.size, "Image and mask sizes do not match"
    assert image_pil.mode == "RGB", "Image must be RGB"
    assert mask_pil.mode == "L", "Mask must be grayscale"

    # convert PIL images to BytesIO
    image_bytes = image_to_bytes(image_pil)
    mask_bytes = image_to_bytes(mask_pil)

    # queue state/create
    stateid_image = await ctx.create_state(file=image_bytes)
    stateid_mask = await ctx.create_state(file=mask_bytes)

    # queue skills/shadow
    stateid_recolor = await ctx.skill_recolor(
        stateid_image=stateid_image,
        stateid_mask=stateid_mask,
        color=params.color,
    )

    # queue state/download
    recolored_pil = await ctx.download_image(stateid_recolor)

    # convert PIL image to tensor
    recolored_tensor = image_to_tensor(recolored_pil).permute(0, 2, 3, 1)

    return recolored_tensor


class Recolor:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "api": (
                    "FG_API",
                    {
                        "tooltip": "The Finegrain API context",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to recolor something in",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask of the object to recolor",
                    },
                ),
                "color": (
                    "STRING",
                    {
                        "default": "#ff0000",
                        "tooltip": "The color to recolor the object to",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Recolor"
    DESCRIPTION = "Recolor a masked object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: torch.Tensor,
        mask: torch.Tensor,
        color: str,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    mask=mask,
                    color=color,
                ),
            ),
        )
