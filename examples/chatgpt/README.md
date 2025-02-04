# ChatGPT bridge

Use the Finegrain API right from ChatGPT.

## Getting started

Create `.env` from the `.env.example` file.

### Development

1. Install the dependencies:
```bash
rye sync --all-features --no-lock
```

2. Start the development server:
```bash
QUART_APP=chatgpt_bridge quart run
```

### Deployment

1. Install the dependencies:
```bash
rye sync --no-dev --no-lock
```

2. Start the production server:
```bash
hypercorn chatgpt_bridge:app
```

3. Create a Custom GPT at <https://chatgpt.com/gpts/mine>.

4. Upload a logo, fill the `Name` and `Description` fields.

5. Fill the `Instructions` field, see `instructions.md` as a reference.

6. Create a new action, see `openapi.yml` as a reference.

7. Set `Authentication` to `API Key - Basic`, and use the `CHATGPT_API_KEY` from the `.env` file.
