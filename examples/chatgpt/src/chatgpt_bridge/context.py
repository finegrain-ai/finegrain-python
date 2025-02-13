import io

from async_lru import alru_cache
from finegrain import EditorAPIContext
from PIL import Image

from chatgpt_bridge.utils import StateID

LRU_SIZE = 16


class EditorAPIContextCached(EditorAPIContext):
    @alru_cache(maxsize=LRU_SIZE)
    async def create_state(
        self,
        file_url: str | None = None,
        file: io.BytesIO | None = None,
        timeout: float | None = None,
    ) -> str:
        assert file_url or file, "file_url or file is required"

        response = await self.request(
            method="POST",
            url="state/create",
            json={"priority": self.priority},
            data={"file_url": file_url} if file_url else None,
            files={"file": file} if file else None,
        )
        state_id = response.json()["state"]
        status = await self.sse_await(state_id, timeout=timeout)
        if status:
            return state_id
        meta = await self.get_meta(state_id)
        raise RuntimeError(f"create_state failed with {state_id}: {meta}")

    @alru_cache(maxsize=LRU_SIZE)
    async def download_image(
        self,
        stateid_image: str,
        image_format: str = "PNG",
        image_resolution: str = "DISPLAY",
    ) -> Image.Image:
        response = await self.request(
            method="GET",
            url=f"state/image/{stateid_image}",
            params={
                "format": image_format,
                "resolution": image_resolution,
            },
        )
        return Image.open(io.BytesIO(response.content))

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_bbox(
        self: EditorAPIContext,
        stateid_image: StateID,
        product_name: str,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"infer-bbox/{stateid_image}",
            params={"product_name": product_name},
        )

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_segment(
        self: EditorAPIContext,
        stateid_bbox: StateID,
    ) -> StateID:
        return await self.ensure_skill(url=f"segment/{stateid_bbox}")

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_cutout(
        self,
        stateid_image: StateID,
        stateid_mask: StateID,
    ) -> StateID:
        return await self.ensure_skill(url=f"cutout/{stateid_image}/{stateid_mask}")

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_merge_masks(
        self,
        stateids: tuple[StateID, ...],
        operation: str = "union",
    ) -> StateID:
        return await self.ensure_skill(
            url="merge-masks",
            params={
                "operation": operation,
                "states": stateids,
            },
        )

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_erase(
        self,
        stateid_image: StateID,
        stateid_mask: StateID,
        mode: str = "free",
    ) -> StateID:
        return await self.ensure_skill(
            url=f"erase/{stateid_image}/{stateid_mask}",
            params={"mode": mode},
        )

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_recolor(
        self,
        stateid_image: StateID,
        stateid_mask: StateID,
        color: str,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"recolor/{stateid_image}/{stateid_mask}",
            params={"color": color},
        )

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_shadow(
        self,
        stateid_cutout: StateID,
        background_color: str,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"shadow/{stateid_cutout}",
            params={"background": background_color},
        )

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_upscale(
        self,
        stateid_image: StateID,
    ) -> StateID:
        return await self.ensure_skill(url=f"upscale/{stateid_image}")

    @alru_cache(maxsize=LRU_SIZE)
    async def skill_set_bgcolor(
        self,
        stateid_image: StateID,
        color: str,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"set-background-color/{stateid_image}",
            params={"background": color},
        )
