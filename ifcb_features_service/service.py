"""Stateless image conversion processor."""

import io
import logging
from typing import List

from fastapi import HTTPException
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field

from stateless_microservice import BaseProcessor, StatelessAction, render_bytes, run_blocking

from ifcb_features.segmentation import segment_roi
from ifcb_features.all import compute_features


logger = logging.getLogger(__name__)


def base64_png_to_array(b64_png_data: str) -> np.ndarray:
    """Convert base64-encoded PNG image data to a numpy array."""
    import base64
    png_data = io.BytesIO(base64.b64decode(b64_png_data))
    with Image.open(png_data) as img:
        arr = np.array(img)
    return arr


def array_to_base64_png(arr: np.ndarray, mode='L') -> str:
    """Convert a numpy array to base64-encoded PNG image data."""
    from PIL import Image
    import base64
    img = Image.fromarray(arr)
    output_io = io.BytesIO()
    img.save(output_io, format='PNG', mode=mode)
    b64_png_data = base64.b64encode(output_io.getvalue()).decode('utf-8')
    return b64_png_data


class BlobRequest(BaseModel):
    """Request payload for segmentation (aka "blob extraction")."""
    image_data: str = Field(
        ...,
        description="Base64-encoded PNG image data",
    )

class FeaturesRequest(BaseModel):
    """Request payload for feature extraction."""
    image_data: str = Field(
        ...,
        description="Base64-encoded PNG image data",
    )

class FeatureProcessor(BaseProcessor):
    """Processor exposing stateless feature extraction."""

    @property
    def name(self) -> str:
        return "feature-extract"

    def get_stateless_actions(self) -> List[StatelessAction]:
        return [
            StatelessAction(
                name="blob-extract",
                path="/blob/extract",
                request_model=BlobRequest,
                handler=self.handle_blob_extraction,
                summary="Compute a blob mask from an IFCB image",
                description="Compute a blob mask from an IFCB image.",
                tags=("blobs",),
            ),
            StatelessAction(
                name="features-extract",
                path="/features/extract",
                request_model=FeaturesRequest,
                handler=self.handle_feature_extraction,
                summary="Extract features from an IFCB image",
                description="Extract features from an IFCB image.",
                tags=("features",),
            ),
        ]

    async def handle_blob_extraction(self, payload: BlobRequest):
        """Compute a blob mask from an IFCB image."""

        def _extract() -> bytes:
            b64_png_data = payload.image_data
            arr = base64_png_to_array(b64_png_data)
            blob = Image.fromarray(segment_roi(arr))
            output_io = io.BytesIO()
            blob.save(output_io, format='PNG', mode='1')
            return output_io.getvalue()

        try:
            blob_png_image_bytes = await run_blocking(_extract)
        except ValueError as exc:
            logger.error("Failed to convert image %s: %s", payload.source_uri, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # convert image to base64-encoded PNG
        return render_bytes(blob_png_image_bytes, 'image/png')

    async def handle_feature_extraction(self, payload: FeaturesRequest):
        """Extract features from an IFCB image."""
        
        def _extract() -> dict:
            b64_png_data = payload.image_data
            # base64 decode
            import base64
            # decode base64 string to bytes
            png_data = io.BytesIO(base64.b64decode(b64_png_data))
            # open image
            with Image.open(png_data) as img:
                arr = np.array(img)
                blob, features = compute_features(arr)
                b64_blob = array_to_base64_png(blob, mode='1')
                return {
                    'blob': b64_blob,
                    'features': dict(features),
                }

        try:
            result = await run_blocking(_extract)
        except ValueError as exc:
            logger.error("Failed to convert image %s: %s", payload.source_uri, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return result