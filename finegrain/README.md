# Finegrain API - Python client

This is a client for the [Finegrain](https://finegrain.ai) API. It requires Python 3.12+ and is designed for asynchronous code using asyncio. It depends on httpx and [httpx_sse](https://github.com/florimondmanca/httpx-sse).

## Setup

We do not publish this package to PyPI or cut releases for now. Here is how to use it with pip:

```bash
pip install "git+https://github.com/finegrain-ai/finegrain-python#subdirectory=finegrain"
```

If you use [uv](https://docs.astral.sh/uv/) (which we recommend) you can do:

```bash
uv add "finegrain @ git+https://github.com/finegrain-ai/finegrain-python#subdirectory=finegrain"
```

## Usage

See [this example script](examples/erase.py) to erase an object from an image by prompt.

If you run in a synchronous context and do not want to manage asyncio, see [this example](examples/erase_sync.py) instead.

## Running tests

You need API credentials (an API key or an email and a password) to run tests. Be careful: doing so will use credits!

To install dependencies, install [Rye](https://rye.astral.sh) and run:

```bash
rye sync
```

The most basic way to run tests is:

```bash
FG_API_CREDENTIALS=FGAPI-ABCDEF-123456-7890AB-CDEF12 rye test
```

If you run tests to debug something you can use for instance:

```bash
odir="/tmp/api-tests-$(date '+%Y-%m-%d-%H-%M-%S')"
mkdir -p "$odir"

FG_API_CREDENTIALS=FGAPI-ABCDEF-123456-7890AB-CDEF12 \
FG_API_URL="https://.../editor" \
FG_TESTS_OUTPUT_DIR="$odir" \
    rye run pytest -v \
    -s -o log_cli=true -o log_level=INFO
```
