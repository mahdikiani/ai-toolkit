"""Asset Manager — store and manage extracted images/charts."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class AssetSourceNotFoundError(FileNotFoundError):
    """Raised when an OCR asset source path does not exist."""

    def __init__(self, source: object) -> None:
        """Initialize the error."""
        super().__init__(f"Asset source not found: {source}")


@dataclass
class Asset:
    """A visual asset (image/chart/figure) extracted from a document."""

    id: str
    original_element_id: str
    filename: str
    path: str
    rel_path: str  # relative to output directory
    width: int
    height: int
    type: str  # "image", "chart", "figure"


class AssetManager:
    """Manages extraction and storage of visual assets."""

    def __init__(self, output_dir: str | Path) -> None:
        """Prepare the assets output directory under *output_dir*."""
        self.output_dir = Path(output_dir)
        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._assets: list[Asset] = []

    def save_image(
        self, image: Image.Image, element_id: str, asset_type: str = "image"
    ) -> Asset:
        """Save an image asset and return Asset record."""
        asset_id = str(uuid.uuid4())[:8]
        filename = f"{asset_type}_{asset_id}.png"
        path = self.assets_dir / filename
        image.save(path, "PNG")

        rel = f"assets/{filename}"
        asset = Asset(
            id=asset_id,
            original_element_id=element_id,
            filename=filename,
            path=str(path),
            rel_path=rel,
            width=image.width,
            height=image.height,
            type=asset_type,
        )
        self._assets.append(asset)
        return asset

    def copy_asset(
        self, source_path: str | Path, element_id: str, asset_type: str = "image"
    ) -> Asset:
        """Copy an existing file as an asset."""
        source = Path(source_path)
        if not source.exists():
            raise AssetSourceNotFoundError(source)
        image = Image.open(source)
        return self.save_image(image, element_id, asset_type)

    def get_assets(self) -> list[Asset]:
        """Return a shallow copy of all recorded assets."""
        return list(self._assets)

    def get_asset_map(self) -> dict[str, str]:
        """Return {element_id: relative_path} mapping."""
        return {a.original_element_id: a.rel_path for a in self._assets}
