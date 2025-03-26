import asyncio
import logging
import re
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from textwrap import dedent
from typing import Any, Literal

import discord
from discord import Intents, app_commands
from environs import Env
from finegrain import DetectResult, EditorAPIContext, ErrorResult, Priority, StateID
from PIL import Image
from typing_extensions import TypeIs

env = Env()
env.read_env()

with env.prefixed("DISCORD_"):
    DISCORD_TOKEN: str | None = env.str("TOKEN")
    DISCORD_GUILD_ID: int | None = env.int("GUILD_ID")
    DISCORD_DEBUG = env.bool("DEBUG", False)
with env.prefixed("FG_"):
    API_URL: str = str(env.str("API_URL", "https://api.finegrain.ai/editor"))
    API_VERIFY: str | bool = env.str("CA_BUNDLE", None) or True
USERS_DB = env.str("USERS_DB", "users.db")
LOGLEVEL = env.str("LOGLEVEL", "INFO").upper()
LOGLEVEL_INT: int = logging.getLevelNamesMapping().get(LOGLEVEL, logging.INFO)

assert DISCORD_GUILD_ID is not None
DISCORD_GUILD = discord.Object(id=DISCORD_GUILD_ID)  # aka "server" in the Discord UI

API_KEY_PATTERN = re.compile(r"^FGAPI(\-[A-Z0-9]{6}){4}$")
ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")
API_PRIORITY: Priority = "standard"
USER_AGENT = "finegrain-discord-bot"
SCISSORS_EMOJI = "\u2702\ufe0f"
INFO_EMOJI = "\u2139\ufe0f"


def init_db() -> None:
    with sqlite3.connect(USERS_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, api_key TEXT NOT NULL)")
        conn.commit()


def add_user(user_id: int, api_key: str) -> None:
    with sqlite3.connect(USERS_DB) as conn:
        conn.execute("INSERT OR REPLACE INTO users (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
        conn.commit()


def maybe_get_user(user_id: int) -> str | None:
    with sqlite3.connect(USERS_DB) as conn:
        cursor = conn.execute("SELECT api_key FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return row[0] if row else None


def remove_user(user_id: int) -> None:
    with sqlite3.connect(USERS_DB) as conn:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()


def is_api_key_valid(api_key: str) -> bool:
    return bool(API_KEY_PATTERN.match(api_key))


def is_error(result: Any) -> TypeIs[ErrorResult]:
    if isinstance(result, ErrorResult):
        raise RuntimeError(result.error)
    return False


@asynccontextmanager
async def get_api_ctx(api_key: str):
    api_ctx = EditorAPIContext(
        api_key=api_key,
        base_url=API_URL,
        priority=API_PRIORITY,
        user_agent=USER_AGENT,
    )

    await api_ctx.login()
    await api_ctx.sse_start()

    yield api_ctx

    await api_ctx.sse_stop()
    api_ctx.token = None  # clear token obtained during login in case of reuse


class UserInputError(app_commands.AppCommandError):
    pass


class ObjectNotFoundError(app_commands.AppCommandError):
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


class BadQualityMaskError(app_commands.AppCommandError):
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


class TooLargeOutputError(app_commands.AppCommandError):
    def __init__(self, actual_size: int, max_size: int) -> None:
        self.actual_size = actual_size
        self.max_size = max_size

    @property
    def actual_size_mb(self) -> str:
        return f"{self.actual_size / (1024 * 1024):.2f} MB"

    @property
    def max_size_mb(self) -> str:
        return f"{self.max_size / (1024 * 1024):.2f} MB"


class FinegrainBot(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.tree.copy_global_to(guild=DISCORD_GUILD)
        await self.tree.sync(guild=DISCORD_GUILD)


intents = Intents.default()
intents.message_content = True

bot = FinegrainBot(intents=intents)

_log = logging.getLogger(__name__)


@dataclass
class BotImage:
    uid: int | str
    content_type: str
    data: bytes
    width: int
    height: int

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

    def compress(self) -> "BotImage":
        with BytesIO(self.data) as f:
            img = Image.open(f)
            with BytesIO() as of:
                img.save(of, format="WEBP", quality=80)
                return BotImage(
                    uid=self.uid,
                    content_type="image/webp",
                    data=of.getvalue(),
                    width=img.width,
                    height=img.height,
                )

    def to_discord_file(self) -> discord.File:
        filename = f"{self.uid}.{self.content_type.split('/')[1]}"
        return discord.File(BytesIO(self.data), filename=filename)


def _get_max_upload_size(interaction: discord.Interaction) -> int | None:
    """Should be 10MB. Except for boosted servers, e.g. 50MB with Level 2 (7 Boosts)."""
    guild = interaction.channel.guild if interaction.channel is not None else None
    return guild.filesize_limit if guild is not None else None


async def _call_segment(
    api_ctx: EditorAPIContext,
    image: BotImage,
    prompt: str,
) -> tuple[StateID, StateID]:
    with BytesIO() as f:
        f.write(image.data)
        st_input = await api_ctx.call_async.upload_image(f)

    segment_r = await api_ctx.call_async.segment(st_input, prompt=prompt, mask_quality="low")
    if isinstance(segment_r, ErrorResult):
        if segment_r.error.startswith("could not identify objects to segment"):
            raise ObjectNotFoundError(prompt=prompt)
        elif segment_r.error.startswith("could not infer a reliable mask"):
            raise BadQualityMaskError(prompt=prompt)
        else:
            raise RuntimeError(segment_r.error)

    return st_input, segment_r.state_id


async def _call_multi_segment(
    api_ctx: EditorAPIContext,
    image: BotImage,
    prompt: str,
) -> tuple[StateID, StateID, DetectResult]:
    with BytesIO() as f:
        f.write(image.data)
        st_input = await api_ctx.call_async.upload_image(f)

    r_detect = await api_ctx.call_async.detect(st_input, prompt=prompt)
    assert not is_error(r_detect)
    if not r_detect.results:
        raise ObjectNotFoundError(prompt=prompt)

    async with asyncio.TaskGroup() as tg:
        bboxes = [target.bbox for target in r_detect.results]
        segmentation_responses = [tg.create_task(api_ctx.call_async.segment(st_input, bbox=bbox)) for bbox in bboxes]

    segmentation_results = [r.result() for r in segmentation_responses]
    if any(isinstance(r, ErrorResult) for r in segmentation_results):
        err = next(r for r in segmentation_results if isinstance(r, ErrorResult))
        raise RuntimeError(err.error)

    st_masks = [r.state_id for r in segmentation_results]
    if len(st_masks) == 1:
        st_mask = st_masks[0]
    else:
        merge_r = await api_ctx.call_async.merge_masks(st_masks, operation="union")
        assert not is_error(merge_r)
        st_mask = merge_r.state_id

    return st_input, st_mask, r_detect


async def _call_object_eraser(
    api_ctx: EditorAPIContext, image: BotImage, prompt: str, mode: Literal["express", "standard", "premium"] = "premium"
) -> BotImage:
    st_input, st_mask = await _call_segment(api_ctx, image, prompt)
    erase_r = await api_ctx.call_async.erase(st_input, st_mask, mode=mode)
    assert not is_error(erase_r)
    image_bytes = await api_ctx.get_image(erase_r.state_id, image_format="JPEG")
    return BotImage(
        uid=erase_r.state_id,
        content_type="image/jpeg",
        data=image_bytes,
        width=erase_r.image_size[0],
        height=erase_r.image_size[1],
    )


async def _call_object_cutter(api_ctx: EditorAPIContext, image: BotImage, prompt: str) -> BotImage:
    st_input, st_mask, _ = await _call_multi_segment(api_ctx, image, prompt)  # needed for high quality masks
    cutout_r = await api_ctx.call_async.cutout(st_input, st_mask)
    assert not is_error(cutout_r)
    cutout_bytes = await api_ctx.get_image(cutout_r.state_id, image_format="PNG")
    cutout = Image.open(BytesIO(cutout_bytes))
    hd_cutout_size = cutout_r.image_size
    left, upper = cutout_r.mask_bbox[:2]

    # Rescaling - should only be needed if DISPLAY resolution is used instead of FULL
    downscale_w, downscale_h = cutout.width / hd_cutout_size[0], cutout.height / hd_cutout_size[1]
    output_size = (int(image.width * downscale_w), int(image.height * downscale_h))
    output_dest = (int(left * downscale_w), int(upper * downscale_h))

    output_image = Image.new("RGBA", output_size, (0, 0, 0, 0))
    output_image.paste(cutout, box=output_dest, mask=cutout)

    with BytesIO() as f:
        output_image.save(f, format="PNG")
        return BotImage(
            uid=cutout_r.state_id,
            content_type="image/png",
            data=f.getvalue(),
            width=output_size[0],
            height=output_size[1],
        )


async def _load_attached_image(attachment: discord.Attachment) -> BotImage:
    if attachment.content_type not in ALLOWED_IMAGE_TYPES:
        raise UserInputError("Please upload a PNG, JPEG or WebP image.")
    return await BotImage.from_attachment(attachment)


def _safe_before_after(
    input_image: BotImage, output_image: BotImage, max_upload_size: int | None = None
) -> tuple[discord.File, discord.File]:
    if max_upload_size is None:
        return input_image.to_discord_file(), output_image.to_discord_file()

    total_size = len(input_image.data) + len(output_image.data)
    if total_size > max_upload_size:
        _log.warning(f"compressing before/after images for output={output_image.uid}")
        input_image, output_image = input_image.compress(), output_image.compress()
        if (final_total_size := len(input_image.data) + len(output_image.data)) > max_upload_size:
            raise TooLargeOutputError(final_total_size, max_upload_size)

    return input_image.to_discord_file(), output_image.to_discord_file()


def is_logged_in(interaction: discord.Interaction) -> bool:
    return maybe_get_user(interaction.user.id) is not None


def is_logged_out(interaction: discord.Interaction) -> bool:
    return not is_logged_in(interaction)


@bot.event
async def on_ready() -> None:
    _log.info(f"logged in as {bot.user}")


@bot.tree.command(name="help")
async def show_help(interaction: discord.Interaction):
    """Show help for the Finegrain Bot."""
    # ruff: noqa: E501
    help_message = dedent(f"""
    **Finegrain Discord bot. \N{SPARKLES} Edit images with prompts!**

    **Basic commands:**
    `/help` - Show this help message
    `/login <api_key>` - Link your Finegrain account to the bot using your `FGAPI-123456-...` API key.
    `/info` - View information about your Finegrain account like remaining credits.
    `/logout` - Unlink your Finegrain account from the bot.

    {INFO_EMOJI} To get an API key, create an account at https://editor.finegrain.ai/signup and go to the "Account settings" page.

    **Image editing commands:**
    `/erase <prompt>` - Remove specific objects from an image, including their shadows and reflections, based on your prompt.
    `/extract <prompt>` - Isolate specific objects by removing the background based on your prompt.

    Example:
    ```
    /erase the potted plant on the left
    /extract the cat on the couch
    ```
    """)
    # ruff: enable
    await interaction.response.send_message(help_message, suppress_embeds=True)


@bot.tree.command(name="login")
@app_commands.describe(
    api_key="Your Finegrain API key created at https://editor.finegrain.ai/.",
)
@app_commands.check(is_logged_out)
async def login(interaction: discord.Interaction, api_key: str) -> None:
    """Link your Finegrain account to the bot."""
    if not is_api_key_valid(api_key):
        await interaction.response.send_message("\N{CROSS MARK} The API key format is invalid.", ephemeral=True)
        return

    api_ctx = EditorAPIContext(
        api_key=api_key,
        base_url=API_URL,
        priority=API_PRIORITY,
        user_agent=USER_AGENT,
    )

    try:
        await api_ctx.login()
    except Exception as e:
        _log.error(f"login failed for {interaction.user.id}", exc_info=e)
        if '"not found"' in str(e):
            reply = "Login failed: API key not found. Please double-check you copied it correctly."
        else:
            reply = "Oops! Something went wrong. Please give it another try in a bit."
        await interaction.response.send_message(reply, ephemeral=True)
        return

    add_user(interaction.user.id, api_key)
    await interaction.response.send_message("\N{WHITE HEAVY CHECK MARK} You are now logged in.", ephemeral=True)


@bot.tree.command(name="info")
@app_commands.check(is_logged_in)
async def info(interaction: discord.Interaction) -> None:
    """View information about your Finegrain account."""
    api_key = maybe_get_user(interaction.user.id)
    assert api_key is not None
    api_ctx = EditorAPIContext(
        api_key=api_key,
        base_url=API_URL,
        priority=API_PRIORITY,
        user_agent=USER_AGENT,
    )

    try:
        await api_ctx.login()
        resp = await api_ctx.request("GET", "auth/me")
        info = resp.json()
    except Exception as e:
        _log.error(f"info failed for {interaction.user.id}", exc_info=e)
        await interaction.response.send_message(
            "Oops! Something went wrong. Please give it another try in a bit.", ephemeral=True
        )
        return

    uid, api_key, num_credits = info["uid"], info["api_key"], info["credits"]
    num_credits = "unlimited" if num_credits == -1 else num_credits
    reply = f"Your info - {interaction.user.mention}\n```User ID: {uid}\nAPI key: {api_key}\nCredits: {num_credits}```"
    await interaction.response.send_message(reply, ephemeral=True)


@bot.tree.command(name="logout")
@app_commands.check(is_logged_in)
async def logout(interaction: discord.Interaction) -> None:
    """Unlink your Finegrain account from the bot."""
    remove_user(interaction.user.id)
    await interaction.response.send_message("You are now logged out.", ephemeral=True)


@bot.tree.command()
@app_commands.rename(attachment="image")
@app_commands.describe(
    attachment="The input image.",
    prompt="Describe the object(s) you want to erase from the image.",
)
@app_commands.check(is_logged_in)
async def erase(
    interaction: discord.Interaction,
    attachment: discord.Attachment,
    prompt: str,
) -> None:
    """Erase one or more objects from the attached image."""
    input_image = await _load_attached_image(attachment)
    interaction.extras["input_file"] = discord.File(BytesIO(input_image.data), filename=attachment.filename)
    await interaction.response.defer(thinking=True)
    api_key = maybe_get_user(interaction.user.id)
    assert api_key is not None
    async with get_api_ctx(api_key) as api_ctx:
        output_image = await _call_object_eraser(api_ctx, input_image, prompt)
    assert output_image.content_type == "image/jpeg"
    reply = f"\N{SPONGE} Before/After for prompt '{prompt}':"
    before_after = _safe_before_after(input_image, output_image, _get_max_upload_size(interaction))
    await interaction.followup.send(reply, files=before_after)


@bot.tree.command()
@app_commands.rename(attachment="image")
@app_commands.describe(
    attachment="The input image.",
    prompt="Describe the object(s) you want to extract from the image.",
)
@app_commands.check(is_logged_in)
async def extract(
    interaction: discord.Interaction,
    attachment: discord.Attachment,
    prompt: str,
) -> None:
    """Extract one or more objects from the attached image."""
    input_image = await _load_attached_image(attachment)
    interaction.extras["input_file"] = discord.File(BytesIO(input_image.data), filename=attachment.filename)
    await interaction.response.defer(thinking=True)
    api_key = maybe_get_user(interaction.user.id)
    assert api_key is not None
    async with get_api_ctx(api_key) as api_ctx:
        output_image = await _call_object_cutter(api_ctx, input_image, prompt)
    assert output_image.content_type == "image/png"
    reply = f"{SCISSORS_EMOJI} Before/After for prompt '{prompt}':"
    before_after = _safe_before_after(input_image, output_image, _get_max_upload_size(interaction))
    await interaction.followup.send(reply, files=before_after)


# NOTE: as per discord.py `ErrorFunc`, the coroutine is annotated with `discord.Interaction` and
# `app_commands.AppCommandError` as arguments. Still, at runtime, this coroutine might receive other kinds of
# exceptions like `ExceptionGroup` from `asyncio.TaskGroup`.
@bot.tree.error
async def on_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    command = interaction.command.name if interaction.command is not None else "N/A"
    _log.error(f"command={command}", exc_info=error)

    reply = "Oops! Something went wrong \N{CONFUSED FACE}. Give it another try in a bit!"
    ephemeral = False
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
    elif isinstance(error, TooLargeOutputError):
        reply = (
            f"Oops! Cannot send the output images because they are too large \N{NO ENTRY SIGN}. "
            f"Maximum allowed size is {error.max_size_mb}, but the output size is {error.actual_size_mb}."
        )
    elif isinstance(error, BadQualityMaskError):
        reply = (
            f"Oops! Could not reliably segment the object(s) in the image based on the prompt '{error.prompt}' "
            "\N{CONFUSED FACE}. "
        )
    elif isinstance(error, app_commands.CheckFailure):
        ephemeral = True
        match command:
            case "login":
                reply = "You are already logged in. Use `/logout` if need be."
            case "logout":
                reply = "You are not logged in, nothing to do."
            case "erase" | "extract":
                reply = "This command requires you to `/login` first."
            case "info":
                reply = "Please `/login` to view your account information."
            case _:
                pass

    if interaction.response.type == discord.InteractionResponseType.deferred_channel_message:
        await interaction.followup.send(reply, files=files, ephemeral=ephemeral)
    else:
        await interaction.response.send_message(reply, files=files, ephemeral=ephemeral)


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
    init_db()
    asyncio.run(start_bot(debug=DISCORD_DEBUG))


if __name__ == "__main__":
    main()
