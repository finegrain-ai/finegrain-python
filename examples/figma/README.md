# figma bridge

Use the Finegrain API right from figma.

## Getting started

Create `.env` from the `.env.example` file.

### Development

1. Install the dependencies:
```bash
uv sync --all-extras --all-groups
```

2. Start the development server:
```bash
QUART_APP=figma_bridge uv run quart
```

### Deployment

1. Install the dependencies:
```bash
uv sync --all-extras --all-groups --locked
```

2. Start the production server:
```bash
hypercorn figma_bridge:app
```
