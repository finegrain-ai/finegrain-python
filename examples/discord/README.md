# finegrain-bot

Easily try out the Finegrain API right from Discord. Built on top of [discord.py](https://github.com/Rapptz/discord.py).

## Prerequisites

1. Install [Rye](https://rye.astral.sh/)
2. Run `rye sync`
3. Copy `.env.example` to `.env` and configure it as you see fit. See `USERS_DB` in particular (SQLite database file)

## Run

Start the bot:

    rye run start

> Note: you can run it from any computer thanks to Discord's [Gateway](https://discord.com/developers/docs/topics/gateway).
