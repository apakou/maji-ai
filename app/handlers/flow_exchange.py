from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from base64 import b64decode, b64encode
from pathlib import Path

import httpx
from cryptography.hazmat.primitives.asymmetric.padding import MGF1, OAEP
from cryptography.hazmat.primitives.asymmetric.padding import hashes as asym_hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from fastapi import APIRouter, Request, Response

from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────
# LOAD PRIVATE KEY (once at startup)
# ─────────────────────────────────────────────


def _load_private_key():
    key_path = Path(os.getenv("ONBOARDING_FLOW_PRIVATE_KEY_PATH", "private_key.pem"))
    raw_passphrase = os.getenv("ONBOARDING_FLOW_PRIVATE_KEY_PASSPHRASE", "")
    passphrase = raw_passphrase.encode() if raw_passphrase else None

    if not key_path.exists():
        raise FileNotFoundError(
            f"Private key not found at {key_path}. "
            "Run: openssl genrsa -out private_key.pem 2048"
        )

    with open(key_path, "rb") as f:
        key_bytes = f.read()

    try:
        private_key = load_pem_private_key(key_bytes, password=passphrase)
    except TypeError as exc:
        # Gracefully handle stale passphrase config with an unencrypted PEM key.
        if passphrase and "private key is not encrypted" in str(exc):
            logger.warning(
                "flow_private_key_load | passphrase provided for unencrypted key, retrying without password"
            )
            private_key = load_pem_private_key(key_bytes, password=None)
        else:
            raise

    logger.info("flow_private_key_loaded | path=%s", key_path)
    return private_key


PRIVATE_KEY = _load_private_key()
APP_SECRET = os.getenv("META_APP_SECRET", "")


# ─────────────────────────────────────────────
# SIGNATURE VALIDATION
# ─────────────────────────────────────────────


def _validate_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """
    Verify the X-Hub-Signature-256 header to confirm the request is from Meta.
    Returns True if valid, False otherwise.
    """
    if not APP_SECRET:
        logger.warning("flow_signature_check | APP_SECRET not set, skipping validation")
        return True  # skip in dev if secret not configured

    if not signature_header or not signature_header.startswith("sha256="):
        logger.warning("flow_signature_check | missing or malformed header")
        return False

    expected = hmac.new(
        APP_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    received = signature_header[len("sha256=") :]

    if not hmac.compare_digest(expected, received):
        logger.error("flow_signature_check | signature mismatch")
        return False

    return True


# ─────────────────────────────────────────────
# DECRYPTION  (Meta spec — data_api_version 3.0)
# ─────────────────────────────────────────────


def decrypt_request(
    encrypted_flow_data_b64: str,
    encrypted_aes_key_b64: str,
    initial_vector_b64: str,
) -> tuple[dict, bytes, bytes]:
    """
    Step 1 — decode all three Base64 fields.
    Step 2 — decrypt the AES key with our RSA private key (OAEP/SHA-256).
    Step 3 — decrypt the flow data with AES-128-GCM.
              The last 16 bytes of the decoded flow data are the GCM auth tag.

    Returns:
        decrypted_data : parsed JSON payload from WhatsApp
        aes_key        : keep this to encrypt your response
        iv             : keep this to build the flipped IV for the response
    """
    flow_data = b64decode(encrypted_flow_data_b64)
    iv = b64decode(initial_vector_b64)
    encrypted_aes = b64decode(encrypted_aes_key_b64)

    # Decrypt AES key using RSA private key
    aes_key = PRIVATE_KEY.decrypt(
        encrypted_aes,
        OAEP(
            mgf=MGF1(algorithm=asym_hashes.SHA256()),
            algorithm=asym_hashes.SHA256(),
            label=None,
        ),
    )

    # Split ciphertext body and GCM auth tag (last 16 bytes)
    encrypted_body = flow_data[:-16]
    auth_tag = flow_data[-16:]

    # Decrypt with AES-128-GCM
    decryptor = Cipher(
        algorithms.AES(aes_key),
        modes.GCM(iv, auth_tag),
    ).decryptor()

    decrypted_bytes = decryptor.update(encrypted_body) + decryptor.finalize()
    decrypted_data = json.loads(decrypted_bytes.decode("utf-8"))

    logger.debug(
        "flow_decrypted | action=%s screen=%s",
        decrypted_data.get("action"),
        decrypted_data.get("screen"),
    )
    return decrypted_data, aes_key, iv


# ─────────────────────────────────────────────
# ENCRYPTION  (Meta spec — data_api_version 3.0)
# ─────────────────────────────────────────────


def encrypt_response(response_data: dict, aes_key: bytes, iv: bytes) -> str:
    """
    Step 1 — flip every bit of the request IV to make the response IV.
    Step 2 — encrypt with AES-128-GCM using the flipped IV.
    Step 3 — append the 16-byte GCM auth tag to the ciphertext.
    Step 4 — Base64-encode the whole thing and return as plain text.
    """
    flipped_iv = bytes(b ^ 0xFF for b in iv)

    encryptor = Cipher(
        algorithms.AES(aes_key),
        modes.GCM(flipped_iv),
    ).encryptor()

    encrypted = (
        encryptor.update(json.dumps(response_data).encode("utf-8"))
        + encryptor.finalize()
        + encryptor.tag  # append the 16-byte auth tag
    )
    return b64encode(encrypted).decode("utf-8")


# ─────────────────────────────────────────────
# HEALTH CHECK  (Meta pings GET on setup)
# ─────────────────────────────────────────────


@router.get("/flow-exchange")
async def flow_health_check():
    """
    Meta sends a GET ping when you save the endpoint URL in Flow Builder.
    Must return 200 — no body required.
    """
    logger.info("flow_health_check_ping")
    return Response(status_code=200)


# ─────────────────────────────────────────────
# MAIN DATA EXCHANGE ENDPOINT
# ─────────────────────────────────────────────


@router.post("/flow-exchange")
async def flow_data_exchange(request: Request):
    """
    Meta calls this endpoint for every screen submission that uses data_exchange.

    Encrypted request body shape:
    {
        "encrypted_flow_data": "<base64>",
        "encrypted_aes_key":   "<base64>",
        "initial_vector":      "<base64>"
    }

    After decryption the payload looks like:
    {
        "version":    "3.0",
        "action":     "data_exchange" | "INIT" | "BACK" | "ping",
        "screen":     "<SCREEN_ID>",
        "data":       { ...form fields from that screen... },
        "flow_token": "<uuid you generated>"
    }

    Response must be the encrypted Base64 string returned as text/plain.
    """
    raw_body = await request.body()

    # ── Validate Meta signature ───────────────────────────────────────────
    signature = request.headers.get("X-Hub-Signature-256")
    if not _validate_signature(raw_body, signature):
        logger.error("flow_exchange | signature validation failed")
        return Response(status_code=432)  # 432 = signature mismatch per Meta spec

    # ── Parse and decrypt ─────────────────────────────────────────────────
    try:
        body = json.loads(raw_body)
        decrypted, aes_key, iv = decrypt_request(
            encrypted_flow_data_b64=body["encrypted_flow_data"],
            encrypted_aes_key_b64=body["encrypted_aes_key"],
            initial_vector_b64=body["initial_vector"],
        )
    except Exception:
        logger.exception("flow_exchange | decryption failed")
        return Response(
            status_code=421
        )  # 421 = cannot decrypt, client re-fetches public key

    action = decrypted.get("action")
    screen = decrypted.get("screen")
    data = decrypted.get("data", {})
    flow_token = decrypted.get("flow_token")

    logger.info(
        "flow_exchange_received | action=%s screen=%s token=%s",
        action,
        screen,
        flow_token,
    )

    # ── Health check ping from Meta ───────────────────────────────────────
    if action == "ping":
        logger.info("flow_exchange | ping received, responding active")
        response_data = {"data": {"status": "active"}}
        return Response(
            content=encrypt_response(response_data, aes_key, iv),
            media_type="text/plain",
        )

    # ── Error notification from client (bad response we sent earlier) ─────
    if action in ("INIT", "data_exchange") and data.get("error"):
        logger.error(
            "flow_exchange_error_notification | screen=%s error=%s message=%s",
            screen,
            data.get("error"),
            data.get("error_message"),
        )
        response_data = {"data": {"acknowledged": True}}
        return Response(
            content=encrypt_response(response_data, aes_key, iv),
            media_type="text/plain",
        )

    # ── Route by screen ───────────────────────────────────────────────────
    try:
        if screen == "PHOTO_CAPTURE":
            response_data = await _handle_photo_screen(data, flow_token)

        elif screen == "INVENTORY_REVIEW":
            response_data = _handle_inventory_screen(data)

        else:
            logger.warning(
                "flow_exchange | unhandled screen=%s action=%s", screen, action
            )
            response_data = _flow_error("Something went wrong. Please try again.")

    except Exception:
        logger.exception("flow_exchange | screen handler error screen=%s", screen)
        response_data = _flow_error(
            "Something went wrong on our end. Please try again."
        )

    # ── Encrypt and return ────────────────────────────────────────────────
    encrypted = encrypt_response(response_data, aes_key, iv)
    logger.debug(
        "flow_exchange_response | screen=%s -> next=%s",
        screen,
        response_data.get("screen"),
    )
    return Response(content=encrypted, media_type="text/plain")


# ─────────────────────────────────────────────
# SCREEN HANDLERS
# ─────────────────────────────────────────────


async def _handle_photo_screen(data: dict, flow_token: str) -> dict:
    """
    User submitted PHOTO_CAPTURE screen.
    Download the image, run Claude Vision, pre-fill INVENTORY_REVIEW.
    """
    from app.handlers.onboarding import step_4_parse_inventory_with_claude

    photo = data.get("photo", {})
    media_id = photo.get("media_id") if isinstance(photo, dict) else None

    if not media_id:
        logger.warning("flow_photo_screen | no media_id | data=%s", data)
        return _flow_error(
            "We could not read your photo. Please try again with a clear, "
            "well-lit picture of your shelves."
        )

    logger.info("flow_photo_screen | downloading media_id=%s", media_id)
    image_b64 = await _download_media_as_base64(media_id)

    if not image_b64:
        return _flow_error("Could not download your photo. Please try again.")

    logger.info("flow_photo_screen | running Claude Vision")
    parsed = await step_4_parse_inventory_with_claude(
        phone=flow_token or "flow_user",
        image_b64s=[image_b64],
        previous_logs=[],
        current_record_strength=0,
        restart_cap=0,
    )
    inventory = parsed.get("inventory", []) if isinstance(parsed, dict) else []

    if not inventory:
        return _flow_error(
            "I could not count the stock from that photo. "
            "Please retake it with better lighting and try again."
        )

    logger.info("flow_photo_screen | parsed items=%d", len(inventory))

    return {
        "screen": "INVENTORY_REVIEW",
        "data": {"inventory": inventory},
    }


async def _download_media_as_base64(media_id: str) -> str | None:
    """Resolve a WhatsApp media URL and return the image bytes as base64 text."""
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    if not access_token:
        logger.error("flow_media_download | missing WHATSAPP_ACCESS_TOKEN")
        return None

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with httpx.AsyncClient(timeout=15) as http:
            url_resp = await http.get(
                f"https://graph.facebook.com/v22.0/{media_id}",
                headers=headers,
            )
        if url_resp.status_code != 200:
            logger.error(
                "flow_media_download | metadata_failed status=%s body=%s",
                url_resp.status_code,
                url_resp.text[:300],
            )
            return None

        media_url = url_resp.json().get("url")
        if not media_url:
            logger.error("flow_media_download | missing media url for media_id=%s", media_id)
            return None

        async with httpx.AsyncClient(timeout=30) as http:
            media_resp = await http.get(media_url, headers=headers)
        if media_resp.status_code != 200:
            logger.error(
                "flow_media_download | download_failed status=%s body=%s",
                media_resp.status_code,
                media_resp.text[:300],
            )
            return None

        return b64encode(media_resp.content).decode("utf-8")
    except Exception:
        logger.exception("flow_media_download | unexpected error media_id=%s", media_id)
        return None


def _handle_inventory_screen(data: dict) -> dict:
    """
    User confirmed/edited inventory and entered stock value + restart cap.
    Calculate their Shield tier and pre-fill CONFIRMATION screen.
    """
    try:
        stock_value = float(
            "".join(filter(str.isdigit, str(data.get("stock_value", 0))))
        )
        restart_cap = float(
            "".join(filter(str.isdigit, str(data.get("restart_cap", 0))))
        )
    except (ValueError, TypeError):
        return _flow_error("Please enter your amounts as numbers only. Example: 50000")

    if restart_cap <= 0:
        return _flow_error(
            "Please enter the amount you would need to restock and reopen "
            "after a flood or fire."
        )

    tier = _calculate_tier(restart_cap)

    logger.info(
        "flow_inventory_screen | stock_value=%s restart_cap=%s tier=%s",
        stock_value,
        restart_cap,
        tier["tier"],
    )

    return {
        "screen": "CONFIRMATION",
        "data": {
            "tier_name": tier["tier"],
            "tier_price": tier["price"],
            "restart_cap_formatted": f"GHS {int(restart_cap):,}",
            "stock_value": stock_value,
            "restart_cap": restart_cap,
        },
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _calculate_tier(restart_cap: float) -> dict:
    if restart_cap <= 5000:
        return {"tier": "Starter Shield", "price": "GHS 60/month"}
    elif restart_cap <= 15000:
        return {"tier": "Standard Shield", "price": "GHS 120/month"}
    else:
        return {"tier": "Premium Shield", "price": "GHS 250/month"}


def _flow_error(message: str) -> dict:
    """
    Per Meta spec, error_message inside the data object shows as a snackbar
    on the current screen. The user stays on that screen and can try again.
    """
    return {
        "data": {
            "error_message": message,
        }
    }
