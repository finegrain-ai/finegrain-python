import io
from typing import Literal

from finegrain import EditorApiAsyncClient as _EditorApiAsyncClient
from finegrain import EditorAPIContext as _EditorAPIContext
from finegrain import ErrorResult, StateID
from PIL import Image


class EditorApiAsyncClient(_EditorApiAsyncClient):
    async def upload_link_image(self, url: str) -> StateID:
        result_create = await self.create_state(file_url=url)
        if isinstance(result_create, ErrorResult):
            raise ValueError(f"Error uploading the link: {result_create.error}")
        return result_create.state_id

    async def download_pil_image(
        self,
        st: StateID,
        image_format: Literal["JPEG", "PNG", "WEBP", "AUTO"] = "PNG",
        resolution: Literal["FULL", "DISPLAY"] = "DISPLAY",
    ) -> Image.Image:
        response = await self.ctx.get_image(state_id=st, image_format=image_format, resolution=resolution)
        return Image.open(io.BytesIO(response))


class EditorAPIContext(_EditorAPIContext):
    @property
    def call_async(self) -> EditorApiAsyncClient:
        return EditorApiAsyncClient(self)
