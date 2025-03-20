import asyncio
import io
from typing import Literal

from finegrain import EditorApiAsyncClient as _EditorApiAsyncClient
from finegrain import EditorAPIContext as _EditorAPIContext
from finegrain import ErrorResult, StateID
from PIL import Image
from quart import current_app as app


class EditorApiAsyncClient(_EditorApiAsyncClient):
    async def upload_link_image(self, url: str) -> StateID:
        result_create = await self.create_state(file_url=url)
        if isinstance(result_create, ErrorResult):
            raise ValueError(f"Error uploading the link: {result_create.error}")
        return result_create.state_id

    async def download_pil_image(
        self,
        st: StateID,
        image_format: Literal["JPEG", "PNG", "WEBP", "AUTO"] = "JPEG",
        resolution: Literal["FULL", "DISPLAY"] = "FULL",
    ) -> Image.Image:
        response = await self.ctx.get_image(state_id=st, image_format=image_format, resolution=resolution)
        return Image.open(io.BytesIO(response))

    async def multi_segment(
        self,
        state_id: StateID,
        prompt: str,
    ) -> StateID:
        # call detect
        result_detect = await self.detect(
            state_id=state_id,
            prompt=prompt,
        )
        if isinstance(result_detect, ErrorResult):
            raise ValueError(f"[multi_segment] internal detect error: {result_detect.error}")
        detections = result_detect.results
        if len(detections) == 0:
            raise ValueError(f"[multi_segment] internal detect error: no detection found for prompt {prompt}")
        detections = detections[:15]  # limit to 15 detections, required by merge-masks
        app.logger.debug(f"{detections=}")

        # call segment on each detection
        async with asyncio.TaskGroup() as tg:
            responses_segment = [
                tg.create_task(
                    self.segment(
                        state_id=state_id,
                        bbox=detection.bbox,
                    )
                )
                for detection in detections
            ]
        results_segment = [r.result() for r in responses_segment]
        if any(isinstance(r, ErrorResult) for r in results_segment):
            err = next(r for r in results_segment if isinstance(r, ErrorResult))
            raise RuntimeError(err.error)
        stateids_segment = [r.state_id for r in results_segment]
        if len(stateids_segment) == 0:
            raise ValueError("[multi_segment] internal segment error: no segmentation found")
        app.logger.debug(f"{stateids_segment=}")

        # call merge-masks
        if len(stateids_segment) == 1:
            stateid_mask_union = stateids_segment[0]
        else:
            result_mask_union = await self.merge_masks(
                state_ids=stateids_segment,
                operation="union",
            )
            if isinstance(result_mask_union, ErrorResult):
                raise ValueError(f"[multi_segment] internal merge_masks error: {result_mask_union.error}")
            stateid_mask_union = result_mask_union.state_id
        app.logger.debug(f"{stateid_mask_union=}")

        return stateid_mask_union


class EditorAPIContext(_EditorAPIContext):
    @property
    def call_async(self) -> EditorApiAsyncClient:
        return EditorApiAsyncClient(self)
