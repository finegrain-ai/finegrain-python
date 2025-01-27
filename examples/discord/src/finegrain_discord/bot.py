import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO
from textwrap import dedent

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
with env.prefixed("FG_"):
    API_URL: str = str(env.str("API_URL", "https://api.finegrain.ai/editor"))
    API_USER: str | None = env.str("API_USER")
    API_PASSWORD: str | None = env.str("API_PASSWORD")
LOGLEVEL = env.str("LOGLEVEL", "INFO").upper()
LOGLEVEL_INT: int = logging.getLevelNamesMapping().get(LOGLEVEL, logging.INFO)

assert DISCORD_GUILD_ID is not None
DISCORD_GUILD = discord.Object(id=DISCORD_GUILD_ID)  # aka "server" in the Discord UI

ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")
API_PRIORITY: Priority = "low"


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
api_ctx = EditorAPIContext(base_url=API_URL, user=API_USER, password=API_PASSWORD, priority=API_PRIORITY)
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


async def _ensure_infer_bbox(api_ctx: EditorAPIContext, state_id: str, prompt: str) -> str:
    url = f"infer-bbox/{state_id}"
    st_boxed, ok = await api_ctx.call_skill(url, {"product_name": prompt})
    if not ok:
        meta_box = await api_ctx.get_meta(st_boxed)
        assert "error" in meta_box
        if meta_box["error"] == "could not infer bbox (not found)":
            raise ObjectNotFoundError(prompt=prompt)
        else:
            raise RuntimeError(f"skill {url} failed with {st_boxed}: {meta_box}")
    return st_boxed


async def _call_object_eraser(api_ctx: EditorAPIContext, image: BotImage, prompt: str) -> BotImage:
    with BytesIO() as f:
        f.write(image.data)
        response = await api_ctx.request("POST", "state/upload", files={"file": f})
    st_input = response.json()["state"]

    st_boxed = await _ensure_infer_bbox(api_ctx, st_input, prompt)
    st_mask = await api_ctx.ensure_skill(f"segment/{st_boxed}")
    st_erased = await api_ctx.ensure_skill(f"erase/{st_input}/{st_mask}", {"mode": "free"})

    response = await api_ctx.request(
        "GET",
        f"state/image/{st_erased}",
        params={"format": "JPEG", "resolution": "DISPLAY"},
    )

    return BotImage(
        uid=st_erased,
        content_type="image/jpeg",
        data=response.content,
    )


async def _call_object_cutter(api_ctx: EditorAPIContext, image: BotImage, prompt: str) -> BotImage:
    assert image.width is not None
    assert image.height is not None
    with BytesIO() as f:
        f.write(image.data)
        response = await api_ctx.request("POST", "state/upload", files={"file": f})
    st_input = response.json()["state"]

    st_boxed = await _ensure_infer_bbox(api_ctx, st_input, prompt)
    st_mask = await api_ctx.ensure_skill(f"segment/{st_boxed}")
    st_cutout = await api_ctx.ensure_skill(f"cutout/{st_input}/{st_mask}")

    response = await api_ctx.request(
        "GET",
        f"state/image/{st_cutout}",
        params={"format": "PNG", "resolution": "DISPLAY"},
    )
    cutout = Image.open(BytesIO(response.content))

    meta_cutout = await api_ctx.get_meta(st_cutout)
    assert "image_size" in meta_cutout
    hd_cutout_size = meta_cutout["image_size"]
    assert "mask_bbox" in meta_cutout
    left, upper = meta_cutout["mask_bbox"][:2]

    # Rescale due to DISPLAY resolution
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
        )


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
    Meet our Discord bot, your friendly Finegrain API playground!

    Inspired by the simplicity of a command line, it's the easiest way to try out Finegrain's magic.

    Just type a slash command, watch it work, and see the results instantly. No setup, no hassle — just pure tinkering fun.

    `/help`
    `/erase`
    `/cutout`

    (and more to come!)

    Just a heads-up: Finegrain offers various speed/quality trade-offs. The bot is optimized for speed, but if you're looking for higher quality, feel free to sign up and get an API key!

    Learn more at https://finegrain.ai
    """)
    # ruff: enable
    await interaction.response.send_message(help_message)


@bot.tree.command()
@app_commands.describe(image="The input image.", prompt="Describe the object you want to erase from the image.")
async def erase(interaction: discord.Interaction, image: discord.Attachment, prompt: str) -> None:
    """Erase a specific object from the attached image."""
    bot_image = await _load_attached_image(image)
    await interaction.response.defer(thinking=True)
    output_bot_image = await _call_object_eraser(bot.api_ctx, bot_image, prompt)
    assert output_bot_image.content_type == "image/jpeg"
    await interaction.followup.send(
        f"Here is your image and the version without '{prompt}'.",
        files=(
            discord.File(BytesIO(bot_image.data), filename=image.filename),
            discord.File(BytesIO(output_bot_image.data), filename=f"{output_bot_image.uid}.jpg"),
        ),
    )


@bot.tree.command()
@app_commands.describe(image="The input image.", prompt="Describe the object you want to cut out from the image.")
async def cutout(interaction: discord.Interaction, image: discord.Attachment, prompt: str) -> None:
    """Create a cutout for a specific object from the attached image."""
    bot_image = await _load_attached_image(image)
    await interaction.response.defer(thinking=True)
    output_bot_image = await _call_object_cutter(bot.api_ctx, bot_image, prompt)
    assert output_bot_image.content_type == "image/png"
    await interaction.followup.send(
        f"Here is your image and the cutout for '{prompt}'",
        files=(
            discord.File(BytesIO(bot_image.data), filename=image.filename),
            discord.File(BytesIO(output_bot_image.data), filename=f"{output_bot_image.uid}.png"),
        ),
    )


@bot.tree.error
async def on_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    command = interaction.command.name if interaction.command is not None else "N/A"
    _log.error(f"command={command}", exc_info=error)

    if isinstance(error, UserInputError):
        reply = f"Oops! That file doesn't work \N{THINKING FACE}. {error}"
    elif isinstance(error, ObjectNotFoundError):
        reply = (
            f"Oops! Couldn't find '{error.prompt}' in the image \N{LEFT-POINTING MAGNIFYING GLASS}. "
            "Try again with a different prompt or image!"
        )
    else:
        reply = "Oops! Something went wrong \N{CONFUSED FACE}. Give it another try in a bit!"

    if interaction.response.type == discord.InteractionResponseType.deferred_channel_message:
        await interaction.followup.send(reply)
    else:
        await interaction.response.send_message(reply)


async def start_bot(reconnect: bool = True) -> None:
    assert DISCORD_TOKEN is not None
    # NOTE: `root=True` means it will affect all loggers rather than just the discord logger, httpx included.
    discord.utils.setup_logging(level=LOGLEVEL_INT, root=True)
    async with bot:
        await bot.start(DISCORD_TOKEN, reconnect=reconnect)


def main() -> None:
    asyncio.run(start_bot())


if __name__ == "__main__":
    main()
