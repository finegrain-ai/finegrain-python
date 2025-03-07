import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO
from textwrap import dedent
from typing import Any, Literal

import discord
from discord import Intents, app_commands
from environs import Env
from finegrain import EditorAPIContext, Priority
from PIL import Image

env = Env()
env.read_env()

with env.prefixed("DISCORD_"):
    DISCORD_TOKEN: str | None = env.str("TOKEN")
    DISCORD_GUILD_ID: int | None = env.int("GUILD_ID")
    DISCORD_DEBUG = env.bool("DEBUG", False)
with env.prefixed("FG_"):
    API_URL: str = str(env.str("API_URL", "https://api.finegrain.ai/editor"))
    API_USER: str | None = env.str("API_USER")
    API_PASSWORD: str | None = env.str("API_PASSWORD")
    API_VERIFY: str | bool = env.str("CA_BUNDLE", None) or True
LOGLEVEL = env.str("LOGLEVEL", "INFO").upper()
LOGLEVEL_INT: int = logging.getLevelNamesMapping().get(LOGLEVEL, logging.INFO)

assert DISCORD_GUILD_ID is not None
DISCORD_GUILD = discord.Object(id=DISCORD_GUILD_ID)  # aka "server" in the Discord UI

ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")
API_PRIORITY: Priority = "standard"
SCISSORS_EMOJI = "\u2702\ufe0f"


class UserInputError(app_commands.AppCommandError):
    pass


class ObjectNotFoundError(app_commands.AppCommandError):
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


class FinegrainBot(discord.Client):
    def __init__(self, api_ctx: EditorAPIContext, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.api_ctx = api_ctx

    async def setup_hook(self):
        self.tree.copy_global_to(guild=DISCORD_GUILD)
        await self.tree.sync(guild=DISCORD_GUILD)
        await self.api_ctx.login()
        await self.api_ctx.sse_start()

    async def close(self) -> None:
        await self.api_ctx.sse_stop()
        await super().close()


intents = Intents.default()
intents.message_content = True

assert API_USER is not None
assert API_PASSWORD is not None
api_ctx = EditorAPIContext(
    base_url=API_URL,
    user=API_USER,
    password=API_PASSWORD,
    priority=API_PRIORITY,
    verify=API_VERIFY,
)
bot = FinegrainBot(api_ctx, intents=intents)

_log = logging.getLogger(__name__)


@dataclass
class BotImage:
    uid: int | str
    content_type: str
    data: bytes
    width: int | None = None
    height: int | None = None

    @classmethod
    async def from_attachment(cls, attachment: discord.Attachment) -> "BotImage":
        assert attachment.content_type in ALLOWED_IMAGE_TYPES
        assert attachment.width is not None
        assert attachment.height is not None
        buffer = BytesIO()
        await attachment.save(buffer)
        return cls(
            uid=attachment.id,
            content_type=attachment.content_type,
            data=buffer.getvalue(),
            width=attachment.width,
            height=attachment.height,
        )


@dataclass
class DetectResult:
    bbox: tuple[int, int, int, int]
    label: str

    @classmethod
    def parse_meta(cls, meta: dict[str, Any]) -> list["DetectResult"]:
        assert "results" in meta
        results: list[DetectResult] = []
        for r in meta["results"]:
            assert "bbox" in r
            assert isinstance(r["bbox"], list)
            assert len(r["bbox"]) == 4
            assert all(isinstance(i, int) for i in r["bbox"])
            assert "label" in r
            assert isinstance(r["label"], str)
            results.append(cls(bbox=tuple(r["bbox"]), label=r["label"]))
        return results

    @classmethod
    def info(cls, results: list["DetectResult"]) -> str:
        # e.g; 1 x 'book', 2 x 'vase'
        return ", ".join(
            f"{len([r for r in results if r.label == label])} x '{label}'" for label in set(r.label for r in results)
        )


async def _call_multi_segment(
    api_ctx: EditorAPIContext, image: BotImage, prompt: str
) -> tuple[str, str, list[DetectResult]]:
    with BytesIO() as f:
        f.write(image.data)
        response = await api_ctx.request("POST", "state/upload", files={"file": f})
    st_input = response.json()["state"]

    st_detect = await api_ctx.ensure_skill(f"detect/{st_input}", {"prompt": prompt})
    meta_detect = await api_ctx.get_meta(st_detect)

    target_objects = DetectResult.parse_meta(meta_detect)
    if not target_objects:
        raise ObjectNotFoundError(prompt=prompt)

    async with asyncio.TaskGroup() as tg:
        segmentations = [
            tg.create_task(api_ctx.ensure_skill(f"segment/{st_input}", {"bbox": target.bbox}))
            for target in target_objects
        ]

    st_masks = [segmentation.result() for segmentation in segmentations]
    if len(st_masks) == 1:
        st_mask = st_masks[0]
    else:
        st_mask = await api_ctx.ensure_skill("merge-masks", {"operation": "union", "states": st_masks})

    return st_input, st_mask, target_objects


async def _call_object_eraser(
    api_ctx: EditorAPIContext, image: BotImage, prompt: str, mode: Literal["express", "standard", "premium"] = "premium"
) -> tuple[BotImage, list[DetectResult]]:
    st_input, st_mask, found_objects = await _call_multi_segment(api_ctx, image, prompt)

    st_erased = await api_ctx.ensure_skill(f"erase/{st_input}/{st_mask}", {"mode": mode})

    response = await api_ctx.request(
        "GET",
        f"state/image/{st_erased}",
        params={"format": "JPEG"},  # FULL resolution by default
    )

    return BotImage(
        uid=st_erased,
        content_type="image/jpeg",
        data=response.content,
    ), found_objects


async def _call_object_cutter(
    api_ctx: EditorAPIContext, image: BotImage, prompt: str
) -> tuple[BotImage, list[DetectResult]]:
    st_input, st_mask, found_objects = await _call_multi_segment(api_ctx, image, prompt)

    st_cutout = await api_ctx.ensure_skill(f"cutout/{st_input}/{st_mask}")

    response = await api_ctx.request(
        "GET",
        f"state/image/{st_cutout}",
        params={"format": "PNG"},  # FULL resolution by default
    )
    cutout = Image.open(BytesIO(response.content))

    meta_cutout = await api_ctx.get_meta(st_cutout)
    assert "image_size" in meta_cutout
    hd_cutout_size = meta_cutout["image_size"]
    assert "mask_bbox" in meta_cutout
    left, upper = meta_cutout["mask_bbox"][:2]

    # Rescaling - should only be needed if DISPLAY resolution is used instead of FULL
    downscale_w, downscale_h = cutout.width / hd_cutout_size[0], cutout.height / hd_cutout_size[1]
    output_size = (int(image.width * downscale_w), int(image.height * downscale_h))
    output_dest = (int(left * downscale_w), int(upper * downscale_h))

    output_image = Image.new("RGBA", output_size, (0, 0, 0, 0))
    output_image.paste(cutout, box=output_dest, mask=cutout)

    with BytesIO() as f:
        output_image.save(f, format="PNG")
        return BotImage(
            uid=st_cutout,
            content_type="image/png",
            data=f.getvalue(),
        ), found_objects


async def _load_attached_image(attachment: discord.Attachment) -> BotImage:
    if attachment.content_type not in ALLOWED_IMAGE_TYPES:
        raise UserInputError("Please upload a PNG, JPEG or WebP image.")
    return await BotImage.from_attachment(attachment)


@bot.event
async def on_ready() -> None:
    _log.info(f"logged in as {bot.user}")


@bot.tree.command(name="help")
async def show_help(interaction: discord.Interaction):
    """Show help for the Finegrain Bot."""
    # ruff: noqa: E501
    help_message = dedent("""
    **Finegrain Discord bot. \N{SPARKLES} Edit images with prompts!**

    `/help` - Show this help message
    `/erase <prompt>` - Remove specific objects from an image, including their shadows and reflections, based on your prompt.
    `/extract <prompt>` - Isolate specific objects by removing the background based on your prompt.

    Example:
    ```
    /erase the potted plant on the left
    /extract the cat on the couch
    ```
    """)
    # ruff: enable
    await interaction.response.send_message(help_message)


@bot.tree.command()
@app_commands.rename(attachment="image")
@app_commands.describe(
    attachment="The input image.",
    prompt="Describe the object(s) you want to erase from the image.",
)
async def erase(
    interaction: discord.Interaction,
    attachment: discord.Attachment,
    prompt: str,
) -> None:
    """Erase one or more objects from the attached image."""
    input_image = await _load_attached_image(attachment)
    interaction.extras["input_file"] = discord.File(BytesIO(input_image.data), filename=attachment.filename)
    await interaction.response.defer(thinking=True)
    output_image, found_objects = await _call_object_eraser(bot.api_ctx, input_image, prompt)
    assert output_image.content_type == "image/jpeg"
    reply = f"\N{SPONGE} Before/After for prompt '{prompt}'. I erased {DetectResult.info(found_objects)}:"
    await interaction.followup.send(
        reply,
        files=(
            interaction.extras["input_file"],
            discord.File(BytesIO(output_image.data), filename=f"{output_image.uid}.jpg"),
        ),
    )


@bot.tree.command()
@app_commands.rename(attachment="image")
@app_commands.describe(
    attachment="The input image.",
    prompt="Describe the object(s) you want to extract from the image.",
)
async def extract(
    interaction: discord.Interaction,
    attachment: discord.Attachment,
    prompt: str,
) -> None:
    """Extract one or more objects from the attached image."""
    input_image = await _load_attached_image(attachment)
    interaction.extras["input_file"] = discord.File(BytesIO(input_image.data), filename=attachment.filename)
    await interaction.response.defer(thinking=True)
    output_image, found_objects = await _call_object_cutter(bot.api_ctx, input_image, prompt)
    assert output_image.content_type == "image/png"
    reply = f"{SCISSORS_EMOJI} Before/After for prompt '{prompt}'. I extracted {DetectResult.info(found_objects)}:"
    await interaction.followup.send(
        reply,
        files=(
            interaction.extras["input_file"],
            discord.File(BytesIO(output_image.data), filename=f"{output_image.uid}.png"),
        ),
    )


# NOTE: as per discord.py `ErrorFunc`, the coroutine is annotated with `discord.Interaction` and
# `app_commands.AppCommandError` as arguments. Still, at runtime, this coroutine might receive other kinds of
# exceptions like `ExceptionGroup` from `asyncio.TaskGroup`.
@bot.tree.error
async def on_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    command = interaction.command.name if interaction.command is not None else "N/A"
    _log.error(f"command={command}", exc_info=error)

    files: list[discord.File] = []
    if isinstance(error, UserInputError):
        reply = f"Oops! That file doesn't work \N{THINKING FACE}. {error}"
    elif isinstance(error, ObjectNotFoundError):
        reply = (
            f"Oops! No objects matching '{error.prompt}' were found in the image \N{LEFT-POINTING MAGNIFYING GLASS}. "
            "Try again with a different prompt or image!"
        )
        input_file = interaction.extras.get("input_file")
        assert isinstance(input_file, discord.File)
        files.append(input_file)
    else:
        reply = "Oops! Something went wrong \N{CONFUSED FACE}. Give it another try in a bit!"

    if interaction.response.type == discord.InteractionResponseType.deferred_channel_message:
        await interaction.followup.send(reply, files=files)
    else:
        await interaction.response.send_message(reply, files=files)


async def start_bot(reconnect: bool = True, debug: bool = False) -> None:
    assert DISCORD_TOKEN is not None
    # NOTE: `root=True` means it will affect all loggers rather than just the discord logger, httpx included.
    discord.utils.setup_logging(level=LOGLEVEL_INT, root=True)
    if debug:
        import aiomonitor

        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

        loop = asyncio.get_running_loop()
        with aiomonitor.start_monitor(loop):
            async with bot:
                await bot.start(DISCORD_TOKEN, reconnect=reconnect)
    else:
        async with bot:
            await bot.start(DISCORD_TOKEN, reconnect=reconnect)


def main() -> None:
    asyncio.run(start_bot(debug=DISCORD_DEBUG))


if __name__ == "__main__":
    main()
