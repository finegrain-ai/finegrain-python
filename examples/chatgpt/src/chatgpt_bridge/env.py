from typing import cast

from environs import Env
from finegrain import Priority

env = Env()
env.read_env()

with env.prefixed("FG_"):
    FG_API_URL: str = str(env.str("API_URL", "https://api.finegrain.ai/editor"))
    FG_API_USER: str = env.str("API_USER")
    FG_API_PASSWORD: str = env.str("API_PASSWORD")
    FG_API_PRIORITY: Priority = cast(Priority, env.str("API_PRIORITY", "low").lower())
    FG_API_TIMEOUT: int = env.int("API_TIMEOUT", 60)

with env.prefixed("CHATGPT_"):
    CHATGPT_AUTH_TOKEN: str = env.str("AUTH_TOKEN")

LOGLEVEL = env.str("LOGLEVEL", "INFO").upper()
APP_LOGLEVEL = env.str("APP_LOGLEVEL", "INFO").upper()
