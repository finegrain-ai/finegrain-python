from importlib.metadata import version
from typing import cast

from environs import Env
from finegrain import Priority

env = Env()
env.read_env()

__version__ = version("chatgpt_bridge")

with env.prefixed("FG_"):
    FG_API_URL: str = str(env.str("API_URL", "https://api.finegrain.ai/editor"))
    FG_API_PRIORITY: Priority = cast(Priority, env.str("API_PRIORITY", "low").lower())
    FG_API_TIMEOUT: int = env.int("API_TIMEOUT", 60)

LOGLEVEL = env.str("LOGLEVEL", "INFO").upper()
APP_LOGLEVEL = env.str("APP_LOGLEVEL", "INFO").upper()

USER_AGENT = f"chatgpt-bridge/{__version__}"
