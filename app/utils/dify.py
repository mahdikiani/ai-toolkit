"""Dify API client utilities for file upload, chat, and OCR."""

import logging
import mimetypes
import os
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image


class DifyClient(httpx.Client):
    """Synchronous HTTP client for the Dify API."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize DifyClient with API key.

        Args:
            api_key: Dify API key. Defaults to DIFY_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
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
        """
        Upload a file to Dify and return the file ID.

        Args:
            file: Path to the file to upload.

        Returns:
            The uploaded file ID from Dify.
        """
        if isinstance(file, str):
            file = Path(file)
        with open(file, "rb") as f:
            files = {"file": (file.name, f, mimetypes.guess_type(file)[0])}
            payload = {"user": "me"}
            response = self.post("/files/upload", data=payload, files=files)
            response.raise_for_status()
            return response.json().get("id")

    def upload_image(self, image: Image.Image) -> str:
        """
        Upload a PIL Image to Dify and return the file ID.

        Args:
            image: PIL Image to upload.

        Returns:
            The uploaded image file ID from Dify.
        """
        with BytesIO() as output:
            image.save(output, format="jpeg")
            output.seek(0)
            files = {"file": ("image.jpg", output, "image/jpeg")}
            payload = {"user": "me"}
            response = self.post("/files/upload", data=payload, files=files)
            response.raise_for_status()
            return response.json().get("id")

    def chat_messages(self, prompt: str, file_id: str | None = None) -> str:
        """
        Send a chat message to Dify and get the response.

        Args:
            prompt: The user's message text.
            file_id: Optional file ID to attach to the message.

        Returns:
            The answer text from Dify.
        """
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
        """
        Perform OCR on an image file using Dify.

        Args:
            file: Path to image file or PIL Image.

        Returns:
            The extracted text from the image.
        """
        if isinstance(file, Image.Image):
            file_id = self.upload_image(file)
        else:
            file_id = self.upload_file(file)
        return self.chat_messages("متن تصویر را بده", file_id)


class AsyncDifyClient(httpx.AsyncClient):
    """Asynchronous HTTP client for the Dify API."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize AsyncDifyClient with API key.

        Args:
            api_key: Dify API key. Defaults to DIFY_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
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
        """
        Upload a file to Dify and return the file ID.

        Args:
            file: Path to the file to upload.

        Returns:
            The uploaded file ID from Dify.
        """
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
        """
        Upload a PIL Image to Dify and return the file ID.

        Args:
            image: PIL Image to upload.

        Returns:
            The uploaded image file ID from Dify.
        """
        with BytesIO() as output:
            image.save(output, format="jpeg")
            output.seek(0)
            return await self.upload_file_bytes(output)

    async def upload_file_bytes(self, file: BytesIO) -> str:
        """
        Upload file bytes to Dify and return the file ID.

        Args:
            file: BytesIO object containing file data.

        Returns:
            The uploaded file ID from Dify.
        """
        file.seek(0)
        files = {"file": ("image.jpg", file.read(), "image/jpeg")}
        payload = {"user": "me"}
        response = await self.post("/files/upload", data=payload, files=files)
        response.raise_for_status()
        return response.json().get("id")

    async def chat_messages(self, prompt: str, file_id: str | None = None) -> str:
        """
        Send a chat message to Dify and get the response.

        Args:
            prompt: The user's message text.
            file_id: Optional file ID to attach to the message.

        Returns:
            The answer text from Dify.
        """
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
        """
        Send a translation request to Dify.

        Args:
            prompt: The text to translate.

        Returns:
            The translated text from Dify.
        """
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
        """
        Perform OCR on an image file using Dify.

        Args:
            file: Path to image file, PIL Image, or BytesIO.

        Returns:
            The extracted text from the image.
        """
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
