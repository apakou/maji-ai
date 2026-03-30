"""
voice_transcriber.py — Download a WhatsApp voice note and transcribe it with Whisper.

Uses OpenAI Whisper (via openai package). Set OPENAI_API_KEY in .env.
WhatsApp sends voice notes as OGG/Opus — Whisper handles these natively.
"""

import logging
import os
import tempfile

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None


async def transcribe_voice_message(media_id: str, phone: str) -> str | None:
    """
    Download a WhatsApp voice note and return its transcript.

    Args:
        media_id: The media_id from the WhatsApp message object.
        phone:    Caller's phone number (used only for logging).

    Returns:
        Transcript string on success, None if transcription is unavailable or fails.
    """
    if _client is None:
        logger.warning("voice_transcribe_skipped | phone=%s reason=no_openai_key", phone)
        return None

    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    try:
        # Step 1 — resolve the download URL
        async with httpx.AsyncClient(timeout=10) as http:
            url_resp = await http.get(
                f"https://graph.facebook.com/v22.0/{media_id}",
                headers=headers,
            )
        media_url = url_resp.json().get("url")

        if not media_url:
            logger.warning(
                "voice_transcribe_no_url | phone=%s media_id=%s", phone, media_id
            )
            return None

        # Step 2 — download the audio bytes
        async with httpx.AsyncClient(timeout=20) as http:
            audio_resp = await http.get(media_url, headers=headers)

        audio_bytes = audio_resp.content
        logger.debug(
            "voice_transcribe_downloaded | phone=%s size_kb=%.1f",
            phone,
            len(audio_bytes) / 1024,
        )

        # Step 3 — write to a temp file and send to Whisper
        # OpenAI SDK needs a file-like object with a name so it can detect format.
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                response = await _client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=None,  # auto-detect (handles English, Twi, Hausa, etc.)
                )
        finally:
            os.unlink(tmp_path)

        transcript = response.text.strip()
        logger.info(
            "voice_transcribe_success | phone=%s chars=%d transcript=%r",
            phone,
            len(transcript),
            transcript[:80],
        )
        return transcript

    except Exception as e:
        logger.exception("voice_transcribe_error | phone=%s error=%s", phone, e)
        return None
