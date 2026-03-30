import logging
import os

import requests
from dotenv.main import load_dotenv

from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()

PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
BASE_URL = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}"
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json",
}


def _post(payload: dict) -> dict:
    """Send any message payload to the WhatsApp API."""
    url = f"{BASE_URL}/messages"
    msg_type = payload.get("type", "unknown")
    to = payload.get("to", "unknown")

    logger.debug("Sending %s message to %s", msg_type, to)

    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            message_id = response_data.get("messages", [{}])[0].get("id", "unknown")
            logger.info(
                "Message sent | type=%s to=%s message_id=%s",
                msg_type,
                to,
                message_id,
            )
        else:
            error = response_data.get("error", {})
            logger.error(
                "Message failed | type=%s to=%s status=%s code=%s message=%s",
                msg_type,
                to,
                response.status_code,
                error.get("code"),
                error.get("message"),
            )

        return response_data

    except requests.exceptions.ConnectionError:
        logger.exception("Network error sending %s message to %s", msg_type, to)
        return {}
    except requests.exceptions.Timeout:
        logger.exception("Timeout sending %s message to %s", msg_type, to)
        return {}
    except Exception:
        logger.exception("Unexpected error sending %s message to %s", msg_type, to)
        return {}


# ─────────────────────────────────────────────
# 1. TYPING INDICATOR / READ RECEIPT
# ─────────────────────────────────────────────


def send_read_receipt(to: str, message_id: str) -> dict:
    """
    Mark a message as read. Shows blue ticks and triggers the typing
    indicator on the user's side. Call this before every reply.

    Args:
        to: User's phone number
        message_id: The 'id' field from the incoming webhook message object
    """
    logger.debug("Sending read receipt to %s for message %s", to, message_id)

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    url = f"{BASE_URL}/messages"

    try:
        response = requests.post(url, headers=HEADERS, json=payload)

        if response.status_code == 200:
            logger.info("Read receipt sent | to=%s message_id=%s", to, message_id)
        else:
            logger.warning(
                "Read receipt failed | to=%s message_id=%s status=%s body=%s",
                to,
                message_id,
                response.status_code,
                response.text,
            )

        return response.json()

    except Exception:
        logger.exception("Error sending read receipt to %s", to)
        return {}


# ─────────────────────────────────────────────
# 2. TEXT MESSAGE
# ─────────────────────────────────────────────


def send_text(to: str, body: str, preview_url: bool = False) -> dict:
    """
    Send a plain text message.
    Supports WhatsApp formatting: *bold*, _italic_, ~strikethrough~, `code`

    Args:
        to: Recipient phone number (e.g. "233501234567")
        body: Message text (max 4096 characters)
        preview_url: Show a link preview if the text contains a URL
    """
    logger.debug("send_text | to=%s chars=%d", to, len(body))

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": body,
            "preview_url": preview_url,
        },
    }
    return _post(payload)


# ─────────────────────────────────────────────
# 3. TEMPLATE MESSAGE
# ─────────────────────────────────────────────


def send_template(
    to: str,
    template_name: str,
    language_code: str = "en_US",
    components: list = None,
) -> dict:
    """
    Send an approved WhatsApp template message.
    Required for first-contact messages outside the 24-hour window.

    Args:
        to: Recipient phone number
        template_name: Exact name of your approved template in Meta Business Manager
        language_code: Language of the template (e.g. "en_US", "en_GB")
        components: List of component objects to fill template variables.

    Example — template with a body variable {{1}}:
        send_template(
            to="233501234567",
            template_name="shop_onboarding_reminder",
            components=[{
                "type": "body",
                "parameters": [{"type": "text", "text": "Ruth"}]
            }]
        )
    """
    logger.debug(
        "send_template | to=%s template=%s lang=%s",
        to,
        template_name,
        language_code,
    )

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
        },
    }
    if components:
        payload["template"]["components"] = components

    return _post(payload)


# ─────────────────────────────────────────────
# 4. INTERACTIVE — REPLY BUTTONS
# ─────────────────────────────────────────────


def send_reply_buttons(
    to: str,
    body_text: str,
    buttons: list[dict],
    header_text: str = None,
    header_image_url: str = None,
    footer_text: str = None,
) -> dict:
    """
    Send a message with up to 3 tappable reply buttons.
    Best for yes/no choices or short option sets.

    Args:
        to: Recipient phone number
        body_text: Main message text
        buttons: List of dicts with 'id' and 'title' keys (max 3, title max 20 chars)
        header_text: Optional bold text above the body
        footer_text: Optional small grey text below the buttons

    Example:
        send_reply_buttons(
            to="233501234567",
            body_text="Does this stock list look correct?",
            buttons=[
                {"id": "confirm_yes", "title": "Yes, looks good ✅"},
                {"id": "confirm_no",  "title": "No, I want to edit"},
            ],
            footer_text="Visbl · Your shop record"
        )
    """
    button_ids = [b["id"] for b in buttons[:3]]
    logger.debug("send_reply_buttons | to=%s button_ids=%s", to, button_ids)

    if len(buttons) > 3:
        logger.warning(
            "send_reply_buttons | to=%s got %d buttons, trimming to 3",
            to,
            len(buttons),
        )

    formatted_buttons = [
        {
            "type": "reply",
            "reply": {"id": btn["id"], "title": btn["title"]},
        }
        for btn in buttons[:3]
    ]

    interactive = {
        "type": "button",
        "body": {"text": body_text},
        "action": {"buttons": formatted_buttons},
    }

    if header_image_url:
        interactive["header"] = {"type": "image", "image": {"link": header_image_url}}
    elif header_text:
        interactive["header"] = {"type": "text", "text": header_text}

    if footer_text:
        interactive["footer"] = {"text": footer_text}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": interactive,
    }
    return _post(payload)


# ─────────────────────────────────────────────
# 5. INTERACTIVE — LIST MESSAGE
# ─────────────────────────────────────────────


def send_list_message(
    to: str,
    body_text: str,
    button_label: str,
    sections: list[dict],
    header_text: str = None,
    footer_text: str = None,
) -> dict:
    """
    Send a message with a scrollable list of options (up to 10 items total).
    Better than buttons when there are more than 3 choices.

    Args:
        to: Recipient phone number
        body_text: Main message text
        button_label: Label on the button that opens the list (max 20 chars)
        sections: List of section dicts. Each section has a 'title' and 'rows'.
                  Each row has 'id', 'title', and optional 'description'.
        header_text: Optional bold text above the body
        footer_text: Optional small grey text below

    Example:
        send_list_message(
            to="233501234567",
            body_text="What type of stock do you mainly sell?",
            button_label="Choose category",
            sections=[{
                "title": "Clothing & Footwear",
                "rows": [
                    {"id": "cat_footwear", "title": "Shoes & Sandals", "description": "All types of footwear"},
                    {"id": "cat_clothing", "title": "Tops & Bottoms",  "description": "Shirts, trousers, dresses"},
                ]
            }]
        )
    """
    total_rows = sum(len(s.get("rows", [])) for s in sections)
    logger.debug(
        "send_list_message | to=%s sections=%d total_rows=%d",
        to,
        len(sections),
        total_rows,
    )

    if total_rows > 10:
        logger.warning(
            "send_list_message | to=%s has %d rows, WhatsApp max is 10",
            to,
            total_rows,
        )

    formatted_sections = []
    for section in sections:
        formatted_rows = [
            {
                "id": row["id"],
                "title": row["title"],
                **(
                    {"description": row["description"]}
                    if row.get("description")
                    else {}
                ),
            }
            for row in section.get("rows", [])
        ]
        formatted_sections.append(
            {
                "title": section.get("title", ""),
                "rows": formatted_rows,
            }
        )

    interactive = {
        "type": "list",
        "body": {"text": body_text},
        "action": {
            "button": button_label,
            "sections": formatted_sections,
        },
    }
    if header_text:
        interactive["header"] = {"type": "text", "text": header_text}
    if footer_text:
        interactive["footer"] = {"text": footer_text}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": interactive,
    }
    return _post(payload)


# ─────────────────────────────────────────────
# 6. IMAGE MESSAGE
# ─────────────────────────────────────────────


def send_image(
    to: str,
    image_url: str = None,
    media_id: str = None,
    caption: str = None,
) -> dict:
    """
    Send an image. Use either a public URL or a pre-uploaded Meta media_id.

    Args:
        to: Recipient phone number
        image_url: Publicly accessible URL of the image (JPEG, PNG, WebP — max 5MB)
        media_id: Meta media ID from a prior upload (preferred for production)
        caption: Optional text shown below the image (max 1024 chars)
    """
    if not image_url and not media_id:
        logger.error(
            "send_image | to=%s called with neither image_url nor media_id", to
        )
        raise ValueError("Provide either image_url or media_id")

    source = f"media_id={media_id}" if media_id else f"url={image_url}"
    logger.debug("send_image | to=%s source=%s", to, source)

    image_obj = {}
    if media_id:
        image_obj["id"] = media_id
    else:
        image_obj["link"] = image_url
    if caption:
        image_obj["caption"] = caption

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": image_obj,
    }
    return _post(payload)


# ─────────────────────────────────────────────
# 7. DOCUMENT MESSAGE
# ─────────────────────────────────────────────


def send_document(
    to: str,
    document_url: str = None,
    media_id: str = None,
    filename: str = "document.pdf",
    caption: str = None,
) -> dict:
    """
    Send a document file (PDF, DOCX, XLSX, etc.).
    Great for sending inventory reports, policy documents, or receipts.

    Args:
        to: Recipient phone number
        document_url: Publicly accessible URL of the document (max 100MB)
        media_id: Meta media ID from a prior upload (preferred)
        filename: The display name shown in the chat (e.g. "Visbl_Report_June.pdf")
        caption: Optional text shown below the document
    """
    if not document_url and not media_id:
        logger.error(
            "send_document | to=%s called with neither document_url nor media_id", to
        )
        raise ValueError("Provide either document_url or media_id")

    source = f"media_id={media_id}" if media_id else f"url={document_url}"
    logger.debug("send_document | to=%s filename=%s source=%s", to, filename, source)

    doc_obj = {"filename": filename}
    if media_id:
        doc_obj["id"] = media_id
    else:
        doc_obj["link"] = document_url
    if caption:
        doc_obj["caption"] = caption

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": doc_obj,
    }
    return _post(payload)


# ─────────────────────────────────────────────
# 8. WHATSAPP FLOW
# ─────────────────────────────────────────────


def send_whatsapp_flow(
    to: str,
    body_text: str,
    flow_id: str,
    flow_cta: str,
    flow_token: str,
    screen: str = None,
    prefill_data: dict = None,
    header_text: str = None,
    header_image_url: str = None,
    footer_text: str = None,
    mode: str = "published",
) -> dict:
    """
    Send a WhatsApp Flow — a native multi-screen form inside WhatsApp.
    Use this for inventory verification, restart cap questions, and onboarding steps.

    Args:
        to: Recipient phone number
        body_text: Message shown before the user opens the Flow
        flow_id: Your Flow ID from Meta Business Manager
        flow_cta: Button label that opens the Flow (e.g. "Review My Records")
        flow_token: A unique token you generate per send (use uuid4)
        screen: The screen to open first (must match a screen ID in your Flow JSON)
        prefill_data: Dict of data to pre-fill form fields
        header_text: Optional bold header above body
        footer_text: Optional footer below CTA button
        mode: "published" for live, "draft" for testing
    """
    logger.debug(
        "send_whatsapp_flow | to=%s flow_id=%s screen=%s mode=%s token=%s",
        to,
        flow_id,
        screen,
        mode,
        flow_token,
    )

    if mode == "draft":
        logger.warning(
            "send_whatsapp_flow | to=%s sending in DRAFT mode — not visible to real users",
            to,
        )

    action_payload = {"screen": screen, "data": prefill_data or {}} if screen else {}

    flow_params = {
        "flow_message_version": "3",
        "flow_token": flow_token,
        "flow_id": flow_id,
        "flow_cta": flow_cta,
        "flow_action": "navigate" if screen else "data_exchange",
        "mode": mode,
    }
    if action_payload:
        flow_params["flow_action_payload"] = action_payload

    interactive = {
        "type": "flow",
        "body": {"text": body_text},
        "action": {
            "name": "flow",
            "parameters": flow_params,
        },
    }
    if header_image_url:
        interactive["header"] = {"type": "image", "image": {"link": header_image_url}}
    elif header_text:
        interactive["header"] = {"type": "text", "text": header_text}

    if footer_text:
        interactive["footer"] = {"text": footer_text}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": interactive,
    }
    return _post(payload)


# ─────────────────────────────────────────────
# TYPING INDICATOR
# ─────────────────────────────────────────────


def send_typing_indicator(to: str, message_id: str) -> dict:
    """
    Show a "typing..." animation in the user's WhatsApp chat.

    This does two things in one API call:
      1. Marks the incoming message as read (blue ticks)
      2. Shows the typing animation for up to 25 seconds

    The animation disappears automatically when you send your reply,
    or after 25 seconds — whichever comes first.

    Args:
        to: User's phone number (e.g. "233501234567")
        message_id: The 'id' field from the incoming webhook message (starts with "wamid.")

    Usage — call this right after receiving a message, before doing any heavy work:
        send_typing_indicator(phone, message_id)
        inventory = await parse_inventory_with_claude(image_b64)  # takes a few seconds
        await send_reply(...)
    """
    logger.debug("send_typing_indicator | to=%s message_id=%s", to, message_id)

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }
    url = f"{BASE_URL}/messages"

    try:
        response = requests.post(url, headers=HEADERS, json=payload)

        if response.status_code == 200:
            logger.info(
                "Typing indicator sent | to=%s message_id=%s",
                to,
                message_id,
            )
        else:
            error = response.json().get("error", {})
            logger.warning(
                "Typing indicator failed | to=%s message_id=%s status=%s code=%s message=%s",
                to,
                message_id,
                response.status_code,
                error.get("code"),
                error.get("message"),
            )

        return response.json()

    except Exception:
        logger.exception("Error sending typing indicator to %s", to)
        return {}
