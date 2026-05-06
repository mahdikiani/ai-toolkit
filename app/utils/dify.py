"""Provide module functionality."""
import logging
import mimetypes
import os
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image


class DifyClient(httpx.Client):
    """Represent DifyClient."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the instance."""
        api_key = api_key or os.getenv("DIFY_API_KEY", "")
        if not api_key or not api_key.strip():
            raise ValueError("DIFY_API_KEY environment variable is not set")

        super().__init__(
            # base_url="https://api.dify.ai/v1",
            base_url="https://api.morshed.pish.run/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300,
        )

    def upload_file(self, file: Path | str) -> str:
        """Run upload file."""
        if isinstance(file, str):
            file = Path(file)
        with open(file, "rb") as f:
            files = {"file": (file.name, f, mimetypes.guess_type(file)[0])}
            payload = {"user": "me"}
            response = self.post("/files/upload", data=payload, files=files)
            response.raise_for_status()
            return response.json().get("id")

    def upload_image(self, image: Image.Image) -> str:
        """Run upload image."""
        with BytesIO() as output:
            image.save(output, format="jpeg")
            output.seek(0)
            files = {"file": ("image.jpg", output, "image/jpeg")}
            payload = {"user": "me"}
            response = self.post("/files/upload", data=payload, files=files)
            response.raise_for_status()
            return response.json().get("id")

    def chat_messages(self, prompt: str, file_id: str | None = None) -> str:
        """Run chat messages."""
        json_data = {
            "inputs": {},
            "query": prompt,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": "me",
            "files": [
                {
                    "type": "image",
                    "transfer_method": "local_file",
                    "upload_file_id": file_id,
                }
            ]
            if file_id
            else [],
        }
        response = self.post("/chat-messages", json=json_data)
        response.raise_for_status()
        return response.json().get("answer")

    def ocr_image(self, file: Path | str | Image.Image) -> str:
        """Run ocr image."""
        if isinstance(file, Image.Image):
            file_id = self.upload_image(file)
        else:
            file_id = self.upload_file(file)
        return self.chat_messages("متن تصویر را بده", file_id)


class AsyncDifyClient(httpx.AsyncClient):
    """Represent AsyncDifyClient."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the instance."""
        api_key = api_key or os.getenv("DIFY_API_KEY", "")
        if not api_key or not api_key.strip():
            raise ValueError("DIFY_API_KEY environment variable is not set")
        super().__init__(
            # base_url="https://api.dify.ai/v1",
            base_url="https://api.morshed.pish.run/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600,
        )

    async def upload_file(self, file: Path | str) -> str:
        """Run upload file."""
        import aiofiles

        if isinstance(file, str):
            file = Path(file)
        async with aiofiles.open(file, "rb") as f:
            files = {"file": (file.name, await f.read(), mimetypes.guess_type(file)[0])}
            payload = {"user": "me"}
            response = await self.post("/files/upload", data=payload, files=files)
            response.raise_for_status()
            return response.json().get("id")

    async def upload_image(self, image: Image.Image) -> str:
        """Run upload image."""
        with BytesIO() as output:
            image.save(output, format="jpeg")
            output.seek(0)
            return await self.upload_file_bytes(output)

    async def upload_file_bytes(self, file: BytesIO) -> str:
        """Run upload file bytes."""
        file.seek(0)
        files = {"file": ("image.jpg", file.read(), "image/jpeg")}
        payload = {"user": "me"}
        response = await self.post("/files/upload", data=payload, files=files)
        response.raise_for_status()
        return response.json().get("id")

    async def chat_messages(self, prompt: str, file_id: str | None = None) -> str:
        """Run chat messages."""
        json_data = {
            "inputs": {},
            "query": prompt,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": "me",
            "files": [
                {
                    "type": "image",
                    "transfer_method": "local_file",
                    "upload_file_id": file_id,
                }
            ]
            if file_id
            else [],
        }
        response = await self.post("/chat-messages", json=json_data)
        response.raise_for_status()
        return response.json().get("answer")

    async def translate(self, prompt: str) -> str:
        """Run translate."""
        json_data = {
            "inputs": {},
            "query": prompt,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": "me",
        }
        response = await self.post("/chat-messages", json=json_data)
        response.raise_for_status()
        return response.json().get("answer")

    async def ocr_image(self, file: Path | str | Image.Image | BytesIO) -> str:
        """Run ocr image."""
        if isinstance(file, Image.Image):
            file_id = await self.upload_image(file)
        elif isinstance(file, BytesIO):
            file_id = await self.upload_file_bytes(file)
        else:
            file_id = await self.upload_file(file)
        return await self.chat_messages("متن تصویر را بده", file_id)


if __name__ == "__main__":
    import dotenv

    logging.basicConfig(level=logging.INFO)
    dotenv.load_dotenv()
    api_key: str = os.getenv("DIFY_API_KEY", "")
    client = DifyClient(api_key)
    image = Path("contents/انتشارات سوره مهر نسخه دیجیتال.jpg")
    text = client.ocr_image(image)
    logging.info(text)
