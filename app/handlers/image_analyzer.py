import base64
import os

import httpx
from dotenv.main import load_dotenv

load_dotenv()
DEEPSEEK_APIKEY = os.getenv("DEEPSEEK_API_KEY")


async def analyze_image_with_deepseek(image_path: str):
    async with httpx.AsyncClient() as client:
        image_data = await client.get(image_path)
        base64_image = base64.b64encode(image_data.content).decode("utf-8")
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_APIKEY}"},
            json={
                "model": "deepseek-vl-7b-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional inventory clerk for market traders. Extract items, categories, and quantities into JSON format only.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this market shop photo and list all visible stock.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                        "response_format": {
                            "type": "json_object",
                        },
                    },
                ],
            },
        )
        return response.json()
