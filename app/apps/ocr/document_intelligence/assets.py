"""Asset Manager — store and manage extracted images/charts."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Asset:
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
        self.output_dir = Path(output_dir)
        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._assets: list[Asset] = []

    def save_image(self, image: Image.Image, element_id: str, type: str = "image") -> Asset:
        """Save an image asset and return Asset record."""
        asset_id = str(uuid.uuid4())[:8]
        filename = f"{type}_{asset_id}.png"
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
            type=type,
        )
        self._assets.append(asset)
        return asset

    def copy_asset(self, source_path: str | Path, element_id: str, type: str = "image") -> Asset:
        """Copy an existing file as an asset."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Asset source not found: {source}")
        image = Image.open(source)
        return self.save_image(image, element_id, type)

    def get_assets(self) -> list[Asset]:
        return list(self._assets)

    def get_asset_map(self) -> dict[str, str]:
        """Return {element_id: relative_path} mapping."""
        return {a.original_element_id: a.rel_path for a in self._assets}
