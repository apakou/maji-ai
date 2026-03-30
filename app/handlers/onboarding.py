import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime

import anthropic
import httpx
from dotenv import load_dotenv

import state
from app.DB.database import SessionLocal, engine
from app.DB.models import (
    Claim,
    InventoryDeclaration,
    InventoryLog,
    Owner,
    Policy,
)
from app.handlers.report_generator import generate_report_pdf, upload_pdf_to_whatsapp
from app.handlers.whatsapp_manager import (
    send_document,
    send_list_message,
    send_reply_buttons,
    send_text,
    send_typing_indicator,
    send_whatsapp_flow,
)
from app.logging_config import setup_logging

load_dotenv()

domain_url = os.getenv("DOMAIN_URL")
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
onboard_flow_id = os.getenv("ONBOARDING_FLOW_ID")

setup_logging()
logger = logging.getLogger(__name__)

_CANCELLABLE_STEPS = {
    "DAILY_AWAITING_INPUT_TYPE",
    "DAILY_AWAITING_PHOTO",
    "DAILY_AWAITING_TEXT",
    "DAILY_AWAITING_CONFIRM",
    "AWAITING_STOCK_UPDATE",
    "AWAITING_INCIDENT_REPORT",
    "AWAITING_REPORT_PAYMENT",
    "AWAITING_CAPS_Q1",
    "AWAITING_CAPS_Q2",
    "AWAITING_FLOW",
    "AWAITING_PHOTO",
    "AWAITING_TEXT_STOCK",
    "AWAITING_INPUT_TYPE",
    "AWAITING_VOICE_STOCK",
}


# ─────────────────────────────────────────────
# STEP 1 — Greeting button
# ─────────────────────────────────────────────
async def step_1_greeting_button(phone: str, name: str = None):
    logger.info("onboarding_step_1_start | phone=%s name=%s", phone, name)

    greeting = f"Hey {name}! 👋" if name else "Hey! 👋"

    send_reply_buttons(
        to=phone,
        body_text=(
            f"Welcome {greeting}.\n\n"
            "Every day you open your store, serve your customers, and work hard for "
            "your family. But at the end of the day — do you know if you’re getting closer to your goals?" 
            "I help you answer that question. No complicated apps. No paperwork. Just WhatsApp."
            "Type Let's go and let’s build something together 💪"
        ),
        buttons=[{"id": "start_onboarding", "title": "Let's go!"}],
        header_image_url=f"{domain_url}/assets/greetings.jpg",
        footer_text="Visbl",
    )

    session_data = {"step": "AWAITING_BUTTON_CLICK"}
    if name:
        session_data["name"] = name

    state.sessions[phone] = session_data
    logger.info("onboarding_step_1_complete | phone=%s", phone)


async def step_1_greeting_interactive_flow(phone: str):
    logger.info("onboarding_step_1_flow_start | phone=%s", phone)
    flow_token = str(uuid.uuid4())
    state.sessions[phone] = {"step": "AWAITING_FLOW", "flow_token": flow_token}
    send_whatsapp_flow(
        to=phone,
        header_image_url=f"{domain_url}/assets/greetings.jpg",
        body_text=(
            "Rainy season is near. "
            "The best time to protect your shop is before anything happens.\n\n"
            "Keep your records safe on your phone. "
            "Protect what you have built."
        ),
        flow_id=onboard_flow_id,
        flow_cta="Show me how",
        flow_token=flow_token,
        screen="WELCOME",
        footer_text="Visbl·",
    )
    logger.info("onboarding_step_1_flow_complete | phone=%s", phone)


# ─────────────────────────────────────────────
# STEP 1b–1f — Profile collection
# ─────────────────────────────────────────────
_CATEGORIES = [
    {"id": "cat_clothing", "title": "Clothing & Footwear"},
    {"id": "cat_food", "title": "Food & Drinks"},
    {"id": "cat_electronics", "title": "Electronics"},
    {"id": "cat_beauty", "title": "Beauty & Cosmetics"},
    {"id": "cat_general", "title": "General Goods"},
    {"id": "cat_other", "title": "Other"},
]
_CATEGORY_LABELS = {c["id"]: c["title"] for c in _CATEGORIES}


async def step_1b_ask_name(phone: str):
    send_text(
        phone,
        "Before we start setting your goals, let me get to know your shop. 🏪\n\n"
        "What is your name?\n\n"
        "_Type *cancel* at any time to go back to the start._",
    )
    state.sessions[phone]["step"] = "AWAITING_NAME"


async def step_1b_skip_name_ask_shop(phone: str):
    """
    Called when we already have the name from WhatsApp profile.
    Skips the 'What is your name?' question and goes straight to shop name.
    """
    send_text(
        phone,
        "Before we start building your record, let me get to know your shop. 🏪\n\n"
        "What is your shop called?\n\n"
        "_Type *cancel* at any time to go back to the start._",
    )
    state.sessions[phone]["step"] = "AWAITING_SHOP"


async def step_1c_handle_name(phone: str, text: str, session: dict):
    name = text.strip().title()
    if not name:
        send_text(phone, "Please tell me your name so I can personalise your record.")
        return
    session["name"] = name
    state.sessions[phone] = session
    send_text(phone, f"Nice to meet you, *{name}*! 👋\n\nWhat is your shop called?")
    state.sessions[phone]["step"] = "AWAITING_SHOP"


async def step_1d_handle_shop(phone: str, text: str, session: dict):
    shop = text.strip().title()
    if not shop:
        send_text(phone, "Please type your shop name to continue.")
        return
    session["shop_name"] = shop
    state.sessions[phone] = session
    send_text(
        phone,
        f"*{shop}* - nice! 🏪\n\n"
        "Where is your shop located?\n\n"
        "Example: Makola Market, Circle, Kumasi Central Market",
    )
    state.sessions[phone]["step"] = "AWAITING_LOCATION"


async def step_1e_handle_location(phone: str, text: str, session: dict):
    location = text.strip().title()
    if not location:
        send_text(phone, "Please type your shop location to continue.")
        return
    session["location"] = location
    state.sessions[phone] = session
    send_list_message(
        to=phone,
        body_text="Last one - what type of products do you mainly sell?",
        button_label="Choose category",
        sections=[
            {
                "title": "Select your category",
                "rows": [{"id": c["id"], "title": c["title"]} for c in _CATEGORIES],
            }
        ],
        footer_text="Visbl · Shop setup",
    )
    state.sessions[phone]["step"] = "AWAITING_CATEGORY"


async def step_1f_handle_category(phone: str, row_id: str, session: dict):
    category = _CATEGORY_LABELS.get(row_id)
    if not category:
        send_list_message(
            to=phone,
            body_text="Please pick your main product category from the list.",
            button_label="Choose category",
            sections=[
                {
                    "title": "Select your category",
                    "rows": [{"id": c["id"], "title": c["title"]} for c in _CATEGORIES],
                }
            ],
        )
        return
    session["category"] = category
    state.sessions[phone] = session
    name = session.get("name", "")
    shop = session.get("shop_name", "your shop")
    send_text(
        phone,
        f"Perfect! *{shop}* is all set up. 🎉\n\n"
        f"Now let's build your stock record, {name}.",
    )
    await step_2_ask_for_photo(phone)


# ─────────────────────────────────────────────
# STEP 2 — Ask how they want to share stock
# ─────────────────────────────────────────────
async def step_2_ask_for_photo(phone: str):
    logger.info("onboarding_step_2_start | phone=%s", phone)
    send_list_message(
        to=phone,
        body_text=(
            "Great! Let's build your shop record.\n\n"
            "How would you like to share your stock?\n\n"
            "_Type *cancel* at any time to go back to the start._"
        ),
        button_label="Choose method",
        sections=[
            {
                "title": "Pick one",
                "rows": [
                    {
                        "id": "input_photo",
                        "title": "📸 Send a photo",
                        "description": "Take a picture of your shelves",
                    },
                    {
                        "id": "input_logbook",
                        "title": "📖 Log book page",
                        "description": "Photo of your written records",
                    },
                    {
                        "id": "input_voice",
                        "title": "🎤 Voice note",
                        "description": "Speak your stock list",
                    },
                    {
                        "id": "input_text",
                        "title": "✍️ Type it out",
                        "description": "Type your list as a message",
                    },
                ],
            }
        ],
        footer_text="Visbl·",
    )
    session = state.sessions.get(phone, {})
    session["step"] = "AWAITING_INPUT_TYPE"
    state.sessions[phone] = session
    logger.info("onboarding_step_2_complete | phone=%s", phone)


async def step_2b_handle_input_choice(phone: str, button_id: str):
    logger.info("onboarding_step_2b | phone=%s choice=%s", phone, button_id)
    session = state.sessions.get(phone, {})
    if button_id in ("input_photo", "input_logbook"):
        label = "your shelves" if button_id == "input_photo" else "your log book page"
        send_text(
            phone,
            f"Take a clear photo of {label} and send it here. 📸\n\n"
            "Make sure everything is in the frame and well lit.",
        )
        session["step"] = "AWAITING_PHOTO"
        state.sessions[phone] = session
    elif button_id == "input_voice":
        send_text(
            phone,
            "Send me a voice note now. 🎤\n\n"
            "Say something like:\n"
            '_"Sneakers 15, Heels 10 at 120 cedis, Bags 5"_\n\n'
            "I will transcribe it and count your stock.",
        )
        session["step"] = "AWAITING_VOICE_STOCK"
        state.sessions[phone] = session
    elif button_id == "input_text":
        send_text(
            phone,
            "Type your stock list below. One item per line or comma-separated.\n\n"
            "Example:\n"
            "Sneakers: 15 @ GHS 120\n"
            "Heels: 10\n"
            "Bags: 5 @ GHS 80\n\n"
            "Add *@ GHS price* after a quantity if you know the unit price. "
            "It is optional.",
        )
        session["step"] = "AWAITING_TEXT_STOCK"
        state.sessions[phone] = session
    else:
        await step_2_ask_for_photo(phone)


async def step_2c_handle_text_stock(phone: str, text: str, session: dict):
    logger.info("onboarding_step_2c_text_stock | phone=%s", phone)
    inventory = await parse_text_inventory(text)
    if not inventory:
        logger.warning("onboarding_step_2c_parse_failed | phone=%s raw=%s", phone, text)
        send_text(
            phone,
            "I couldn't read that list. Please use this format:\n\n"
            "Sneakers: 15\nHeels: 10\nBags: 5\n\n"
            "Or: Sneakers 15, Heels 10, Bags 5",
        )
        return
    session["inventory"] = inventory
    state.sessions[phone] = session
    logger.info(
        "onboarding_step_2c_complete | phone=%s items=%d", phone, len(inventory)
    )
    await step_4_trigger_verification(phone, inventory)


# ─────────────────────────────────────────────
# STEP 3 — Receive photo + download from Meta
# ─────────────────────────────────────────────
async def step_3_handle_photo(phone: str, message: dict, session: dict):
    # ── BUFFERING LOGIC ──
    logger.info("onboarding_photo_buffer | phone=%s", phone)
    live = state.sessions.get(phone, {})

    # If we are already processing, ignore (concurrent check in main.py handles this too)
    if live.get("step") == "PROCESSING_PHOTO":
        return

    if "image_buffer" not in live:
        live["image_buffer"] = []

    live["image_buffer"].append(message)

    # Set debounce deadline (2.0s from now)
    deadline = time.time() + 2.0
    live["buffer_deadline"] = deadline
    state.sessions[phone] = live

    # Launch background waiter
    asyncio.create_task(_step_3_buffer_waiter(phone, deadline))


async def _step_3_buffer_waiter(phone, task_deadline):
    # Wait slightly longer than the debounce window
    await asyncio.sleep(2.5)

    live = state.sessions.get(phone, {})
    current_deadline = live.get("buffer_deadline", 0)

    # If deadline advanced, a newer task will handle it
    if current_deadline > task_deadline:
        return

    messages = live.get("image_buffer", [])
    if not messages:
        return

    # Lock state and clear buffer
    live["step"] = "PROCESSING_PHOTO"
    live["image_buffer"] = []
    live.pop("buffer_deadline", None)
    state.sessions[phone] = live

    await _step_3_process_images(phone, messages)


async def _step_3_process_images(phone: str, messages: list):
    t0 = time.perf_counter()
    logger.info("onboarding_step_3_start | phone=%s count=%d", phone, len(messages))

    # Use the last message for reply context
    last_message = messages[-1]
    message_id = last_message["id"]

    send_text(phone, "Got it! Counting your stock now... ⏳")
    send_typing_indicator(phone, message_id)

    image_b64s = []

    for message in messages:
        media_ids = (
            [message["image"]]
            if isinstance(message["image"], dict)
            else message["image"]
        )
        for media_id_obj in media_ids:
            media_id = media_id_obj["id"]
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    url_resp = await client.get(
                        f"https://graph.facebook.com/v22.0/{media_id}",
                        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                    )
                media_url = url_resp.json().get("url")
            except Exception:
                logger.exception(
                    "onboarding_step_3_media_url_error | phone=%s media_id=%s",
                    phone,
                    media_id,
                )
                continue

            if not media_url:
                continue

            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    img_resp = await client.get(
                        media_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
                    )
                image_b64s.append(base64.b64encode(img_resp.content).decode("utf-8"))
            except Exception:
                logger.exception("onboarding_step_3_download_error | phone=%s", phone)
                continue

    if not image_b64s:
        state.sessions[phone]["step"] = "AWAITING_PHOTO"
        send_text(
            phone,
            "I could not download your photos. Please try again. 📸",
        )
        return

    # ── CHANGED: pull owner context before calling Claude ──
    previous_logs, record_strength, restart_cap = _get_owner_context(phone)

    result = await step_4_parse_inventory_with_claude(
        phone,
        image_b64s,
        previous_logs=previous_logs,
        current_record_strength=record_strength,
        restart_cap=restart_cap,
    )

    inventory = result.get("inventory", [])
    if not inventory:
        logger.warning("onboarding_step_3_no_inventory | phone=%s", phone)
        state.sessions[phone]["step"] = "AWAITING_PHOTO"
        send_text(
            phone,
            "I could not read the stock clearly from those photos. "
            "Please try again with clearer, well-lit pictures. 📸",
        )
        return

    # Store record strength in session
    session = state.sessions.get(phone, {})
    session["record_strength"] = result.get("record_strength_score", 0)
    session["inventory"] = inventory
    state.sessions[phone] = session

    logger.info(
        "onboarding_step_3_complete | phone=%s items=%d elapsed=%.2fs",
        phone,
        len(inventory),
        time.perf_counter() - t0,
    )
    # ── CHANGED: pass audit result to trigger verification ──
    await step_4_trigger_verification(phone, inventory, audit_result=result)


async def step_3_handle_voice(phone: str, message: dict, session: dict):
    """Onboarding: transcribe a voice note then hand off to text stock parser."""
    t0 = time.perf_counter()
    logger.info("onboarding_step_3_voice_start | phone=%s", phone)
    message_id = message["id"]
    media_id = message.get("audio", {}).get("id")
    if not media_id:
        send_reply_buttons(
            to=phone,
            body_text="I could not read that voice note. Please try again. 🎤",
            buttons=[
                {"id": "input_voice", "title": "🎤 Try again"},
                {"id": "input_text", "title": "✍️ Type instead"},
                {"id": "input_photo", "title": "📸 Send a photo"},
            ],
            footer_text="Visbl·",
        )
        session["step"] = "AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url_resp = await client.get(
                f"https://graph.facebook.com/v22.0/{media_id}",
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            )
        media_url = url_resp.json().get("url")
    except Exception:
        logger.exception("onboarding_step_3_voice_url_error | phone=%s", phone)
        send_reply_buttons(
            to=phone,
            body_text="I could not reach WhatsApp to download your voice note. Please try again. 🎤",
            buttons=[
                {"id": "input_voice", "title": "🎤 Try again"},
                {"id": "input_text", "title": "✍️ Type instead"},
                {"id": "input_photo", "title": "📸 Send a photo"},
            ],
            footer_text="Visbl·",
        )
        session["step"] = "AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    if not media_url:
        send_reply_buttons(
            to=phone,
            body_text="I could not read that voice note. Please try again. 🎤",
            buttons=[
                {"id": "input_voice", "title": "🎤 Try again"},
                {"id": "input_text", "title": "✍️ Type instead"},
                {"id": "input_photo", "title": "📸 Send a photo"},
            ],
            footer_text="Visbl·",
        )
        session["step"] = "AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            audio_resp, _ = await asyncio.gather(
                client.get(
                    media_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
                ),
                asyncio.to_thread(
                    send_text, phone, "Got it! Listening to your stock list... 🎤⏳"
                ),
            )
    except Exception:
        logger.exception("onboarding_step_3_voice_download_error | phone=%s", phone)
        send_reply_buttons(
            to=phone,
            body_text="I had trouble downloading your voice note. Please try again. 🎤",
            buttons=[
                {"id": "input_voice", "title": "🎤 Try again"},
                {"id": "input_text", "title": "✍️ Type instead"},
                {"id": "input_photo", "title": "📸 Send a photo"},
            ],
            footer_text="Visbl·",
        )
        session["step"] = "AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    send_typing_indicator(phone, message_id)
    transcript = await _transcribe_audio(phone, audio_resp.content)
    if not transcript:
        logger.warning("onboarding_step_3_voice_transcription_failed | phone=%s", phone)
        send_reply_buttons(
            to=phone,
            body_text=(
                "I could not understand that voice note. 🎤\n\n"
                "Speak clearly and say items like:\n"
                '_"Sneakers 15, Heels 10 at 120 cedis, Bags 5"_'
            ),
            buttons=[
                {"id": "input_voice", "title": "🎤 Try again"},
                {"id": "input_text", "title": "✍️ Type instead"},
                {"id": "input_photo", "title": "📸 Send a photo"},
            ],
            footer_text="Visbl·",
        )
        session["step"] = "AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    logger.info(
        "onboarding_step_3_voice_transcribed | phone=%s transcript=%s elapsed=%.2fs",
        phone,
        transcript,
        time.perf_counter() - t0,
    )
    inventory = await parse_text_inventory(transcript)
    if not inventory:
        logger.warning("onboarding_step_3_voice_parse_failed | phone=%s", phone)
        send_reply_buttons(
            to=phone,
            body_text=(
                "I heard you but could not read a stock list from that. 🎤\n\n"
                'Try saying: _"Sneakers 15, Heels 10, Bags 5"_'
            ),
            buttons=[
                {"id": "input_voice", "title": "🎤 Try again"},
                {"id": "input_text", "title": "✍️ Type instead"},
                {"id": "input_photo", "title": "📸 Send a photo"},
            ],
            footer_text="Visbl·",
        )
        session["step"] = "AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    session["inventory"] = inventory
    state.sessions[phone] = session
    await step_4_trigger_verification(phone, inventory)


# ─────────────────────────────────────────────
# CONTEXT HELPER — pulls owner logs from DB
# ── NEW FUNCTION ──
# ─────────────────────────────────────────────
def _get_owner_context(phone: str) -> tuple:
    """
    Pull the last 5 logs for this owner from the database.
    Separates stock snapshots from sales so Claude can reason
    about whether stock movement is explained by reported sales.

    Returns:
        previous_logs (list): formatted log entries for Claude
        record_strength (int): current strength score
        restart_cap (float): owner's stated restart cap
    """
    db = SessionLocal()
    try:
        owner = db.query(Owner).filter(Owner.phone_number == phone).first()

        if not owner:
            return [], 0, 0

        record_strength = getattr(owner, "record_strength", 0) or 0

        # Pull restart cap from most recent policy
        policy = (
            db.query(Policy)
            .filter(Policy.owner_id == owner.id)
            .order_by(Policy.id.desc())
            .first()
        )
        restart_cap = policy.payout_cap_pesewas / 100 if policy else 0

        # Pull last 5 inventory log entries
        logs = (
            db.query(InventoryLog)
            .filter(InventoryLog.owner_id == owner.id)
            .order_by(InventoryLog.logged_at.desc())
            .limit(5)
            .all()
        )

        previous_logs = []
        for log in logs:
            entry = {
                "date": log.logged_at.strftime("%d %b %Y"),
                "type": log.entry_type or "unknown",
                "item": log.product_name or "unknown item",
                "quantity": log.quantity or 0,
            }
            if log.unit_price_pesewas:
                entry["unit_price_ghs"] = log.unit_price_pesewas / 100
            previous_logs.append(entry)

        return previous_logs, record_strength, restart_cap

    except Exception:
        logger.exception("get_owner_context_error | phone=%s", phone)
        return [], 0, 0
    finally:
        db.close()


# ─────────────────────────────────────────────
# STEP 4 — Claude Vision + Audit Engine
# ── CHANGED: returns dict instead of list ──
# ── CHANGED: context-aware mismatch logic ──
# ── CHANGED: uses Sonnet for full audit ──
# ─────────────────────────────────────────────
async def step_4_parse_inventory_with_claude(
    phone: str,
    image_b64s: list,
    previous_logs: list = None,
    current_record_strength: int = 0,
    restart_cap: float = 0,
) -> dict:
    """
    Upgraded Claude vision call.

    Returns dict with:
      - inventory: list of items
      - estimated_total_value_ghs: int
      - record_strength_score: int
      - record_strength_change: int
      - verification_status: match | mismatch | unverified
      - risk_flag: none | low | high
      - insight: one actionable sentence
      - user_message: full receipt string for WhatsApp

    Previous version returned a plain list.
    This version returns a dict with audit fields.
    """
    t0 = time.perf_counter()
    logger.info("claude_vision_start | phone=%s", phone)

    # ── Format previous logs for Claude ──
    # Separate stock snapshots from sales so Claude can reason
    # about whether stock movement is explained by reported sales.
    logs_context = "No previous logs yet. This is the first entry."
    if previous_logs:
        snapshot_lines = []
        sale_lines = []
        for log in previous_logs:
            entry_type = log.get("type", "unknown")
            item = log.get("item", "unknown")
            qty = log.get("quantity", 0)
            date = log.get("date", "unknown date")
            price = log.get("unit_price_ghs")

            if entry_type == "sale":
                line = f"  {date}: SOLD {qty}x {item}"
                if price:
                    line += f" at GHS {price:.0f} each"
                sale_lines.append(line)
            else:
                line = f"  {date}: STOCK SNAPSHOT — {qty}x {item}"
                snapshot_lines.append(line)

        parts = []
        if snapshot_lines:
            parts.append("Previous stock snapshots:\n" + "\n".join(snapshot_lines))
        if sale_lines:
            parts.append(
                "Sales recorded since last snapshot:\n" + "\n".join(sale_lines)
            )
        if parts:
            logs_context = "\n\n".join(parts)

    system_prompt = f"""You are the Visbl Business Record Engine.
You help informal container shop traders in Ghana build
verified digital inventory records for disaster protection.

You know Circle market, Makola, and Osu well.
You understand how traders work — waybills, stock runs,
MoMo payments, SUSU groups, market days.

━━━━━━━━━━━━━━━━━━
TRADER CONTEXT
━━━━━━━━━━━━━━━━━━
{logs_context}

Current record strength: {current_record_strength}/100
Restart cap: GHS {restart_cap:,.0f}

━━━━━━━━━━━━━━━━━━
YOUR INSTRUCTIONS
━━━━━━━━━━━━━━━━━━

1. IDENTIFY STOCK
   A. SHELF/SHOP PHOTO:
      List all visible product categories and estimate quantities.
      Estimate unit price in GHS if visible.

   B. LOGBOOK/PAPER RECORD:
      Transcribe the written items exactly.
      Extract quantity and price for each line.
      If quantity is missing/unclear, default to 1.
      If the page shows sales, capture the items and quantities as listed.

   C. GENERAL:
      Include date only if clearly written. Never guess dates.

2. COMPARE AGAINST PREVIOUS LOGS
   Stock naturally goes down when items are sold.
   A drop in quantity is NOT a mismatch on its own.

   Flag a mismatch ONLY when:
   - Stock dropped and NO sales were logged to
     explain the drop
   - New items appear with no restock logged
   - Stock increased with no restock logged

   If sales in the log explain the stock movement,
   set verification_status to "match" and note it
   in the user_message.

   Example — NOT a mismatch:
   Log shows: 50 sneakers snapshot, then 38 sales
   Photo shows: ~12 sneakers
   Result: verification_status = "match"
   Note: "Stock matches your sales records."

   Example — IS a mismatch:
   Log shows: 50 sneakers snapshot, no sales logged
   Photo shows: ~10 sneakers
   Result: verification_status = "mismatch"
   Note: Ask trader what happened to the stock.

   If there are no previous logs, set
   verification_status to "unverified" — this is
   the first entry, nothing to compare against.

3. RECORD STRENGTH SCORING
   Start from current score: {current_record_strength}
   Award points:
   +5  photo provided
   +10 photo matches previous log (verification match)
   +5  price visible in image
   +5  date visible in image
   Deduct points:
   -15 verification mismatch
   -5  no previous logs (first entry, normal)
   Cap final score between 0 and 100.

4. VALUE ESTIMATION
   Use average GHS market prices for Ghana informal
   retail (Circle market, Makola, Osu range).
   Always label as ESTIMATED in the user_message.
   Never present an estimate as a verified valuation.

5. INSIGHT
   One short actionable sentence for the trader.
   Examples:
   "Add price tags to your shelves — it strengthens
   your insurance record."
   "Log your sales daily to keep your record strong."
   "Your record strength is growing — keep it up."

6. USER MESSAGE — THE DIGITAL RECEIPT
   Format the user_message as a WhatsApp receipt.
   Warm tone. Short sentences. Market language.
   Use: Shop, Stock, MoMo, Waybill.
   Include the receipt format shown below.

━━━━━━━━━━━━━━━━━━
CRITICAL: Every JSON key must be in double quotes. Never write unquoted keys like  qty: 1 — always write \"qty\": 1.
No markdown. No backticks. Raw JSON only.
━━━━━━━━━━━━━━━━━━

{{
  "inventory": [
    {{
      "item": "string",
      "qty": number,
      "price": number or null,
      "date": "string or null"
    }}
  ],
  "estimated_total_value_ghs": number,
  "record_strength_score": number,
  "record_strength_change": number,
  "verification_status": "match/mismatch/unverified",
  "risk_flag": "none/low/high",
  "insight": "one actionable sentence",
  "user_message": "full receipt string"
}}"""

    raw = None
    try:
        claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        image_blocks = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            }
            for img_b64 in image_b64s
        ]

        response = await asyncio.wait_for(
            claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,  # Increased from 1024 to prevent truncation
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            *image_blocks,
                            {
                                "type": "text",
                                "text": (
                                    "Analyse this shop photo. "
                                    "Compare against the previous logs above. "
                                    "Return JSON only."
                                ),
                            },
                        ],
                    }
                ],
            ),
            timeout=45.0,  # Increased timeout for longer generation
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                "claude_vision_json_repair_attempt | phone=%s error=%s",
                phone,
                e,
            )
            try:
                repaired = _repair_json(raw)
                result = json.loads(repaired)
                logger.info("claude_vision_json_repaired | phone=%s", phone)
            except json.JSONDecodeError as e2:
                logger.error(
                    "claude_vision_json_error | phone=%s error=%s raw=%s",
                    phone,
                    e2,
                    raw,
                )
                return {}

        logger.info(
            "claude_vision_complete | phone=%s items=%d "
            "strength=%d status=%s elapsed=%.2fs",
            phone,
            len(result.get("inventory", [])),
            result.get("record_strength_score", 0),
            result.get("verification_status", "unknown"),
            time.perf_counter() - t0,
        )
        return result

    except asyncio.TimeoutError:
        logger.error(
            "claude_vision_timeout | phone=%s elapsed=%.2fs",
            phone,
            time.perf_counter() - t0,
        )
        return {}
    except json.JSONDecodeError as e:
        logger.error(
            "claude_vision_json_error | phone=%s error=%s raw=%s",
            phone,
            e,
            raw,
        )
        return {}
    except Exception:
        logger.exception(
            "claude_vision_error | phone=%s elapsed=%.2fs",
            phone,
            time.perf_counter() - t0,
        )
        return {}


async def step_4_trigger_verification(
    phone: str,
    inventory: list,
    audit_result: dict = None,
):
    logger.info(
        "onboarding_step_4_verification_trigger | phone=%s items=%d",
        phone,
        len(inventory),
    )

    # ─────────────────────────────────────────────
    # Extract audit fields
    # ─────────────────────────────────────────────
    total_value = 0
    strength = 0
    insight = ""
    verification_status = "unverified"

    if audit_result:
        total_value = audit_result.get("estimated_total_value_ghs", 0)
        strength = audit_result.get("record_strength_score", 0)
        insight = audit_result.get("insight", "")
        verification_status = audit_result.get("verification_status", "unverified")

        # Persist record strength in session
        session = state.sessions.get(phone, {})
        session["record_strength"] = strength
        state.sessions[phone] = session

    # ─────────────────────────────────────────────
    # Format item lines for the plain text message
    # ─────────────────────────────────────────────
    def _fmt_item(i: dict) -> str:
        line = f"• {i['item']}: {i['qty']} pieces"
        if i.get("price"):
            line += f" @ GHS {i['price']:,.0f}"
        if i.get("date"):
            line += f" (logged {i['date']})"
        return line

    items_text = "\n".join([_fmt_item(i) for i in inventory])

    # ─────────────────────────────────────────────
    # MESSAGE 1 — plain text with full items list
    # Sent separately so it never hits the 1024
    # char limit on interactive messages
    # ─────────────────────────────────────────────
    send_text(phone, f"Here is what I recorded from your shop:\n\n{items_text}")
    logger.info(
        "onboarding_step_4_items_text_sent | phone=%s items=%d",
        phone,
        len(inventory),
    )

    # ─────────────────────────────────────────────
    # Verification badge
    # ─────────────────────────────────────────────
    if verification_status == "match":
        badge = "✅ Matches your records"
    elif verification_status == "mismatch":
        badge = "⚠️ Something looks different"
    else:
        badge = "📋 Recorded by Visbl AI"

    # ─────────────────────────────────────────────
    # Build short summary — kept well under 1024
    # chars for the interactive message body
    # ─────────────────────────────────────────────
    summary = (
        f"Visbl RECEIPT 🧾\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Date: {datetime.now().strftime('%d %b %Y')}\n"
        f"Est. Value: GHS {total_value:,}\n"
        f"Record Strength: {strength}/100\n"
        f"{badge}"
    )

    if insight:
        summary += f"\n\n💡 {insight}"

    if verification_status == "mismatch":
        summary += (
            "\n\n⚠️ Your stock looks lower than your last record. "
            "If you sold items, reply with how many were sold "
            "so I can update your records."
        )

    # ─────────────────────────────────────────────
    # MESSAGE 2 — interactive button/flow message
    # body_text hard-capped at 1024 chars as a
    # safety net even though summary is short
    # ─────────────────────────────────────────────
    MAX_BODY = 1024

    flow_id = os.getenv("WHATSAPP_FLOW_ID")

    if flow_id:
        flow_token = str(uuid.uuid4())
        state.sessions[phone]["flow_token"] = flow_token

        body_text = f"{summary}\n\nPlease confirm or correct below."
        body_text = body_text[:MAX_BODY]

        send_whatsapp_flow(
            to=phone,
            header_text="Your Shop Record",
            body_text=body_text,
            flow_id=flow_id,
            flow_cta="Review My Records",
            flow_token=flow_token,
            screen="INVENTORY_REVIEW",
            prefill_data={"inventory": inventory},
            footer_text="Visbl·",
        )
        logger.info("onboarding_step_4_flow_sent | phone=%s", phone)

    else:
        body_text = f"{summary}\n\nDoes this look correct?"
        body_text = body_text[:MAX_BODY]

        send_reply_buttons(
            to=phone,
            header_text="Your Shop Record",
            body_text=body_text,
            buttons=[
                {"id": "inventory_correct", "title": "Yes, correct ✅"},
                {"id": "inventory_edit", "title": "No, edit it"},
            ],
            footer_text="Visbl·",
        )
        logger.info("onboarding_step_4_buttons_sent | phone=%s", phone)

    state.sessions[phone]["step"] = "AWAITING_FLOW"


# ─────────────────────────────────────────────
# STEP 4b — Handle Natural Language Correction
# ─────────────────────────────────────────────
async def step_4b_handle_correction(phone: str, text: str, session: dict):
    logger.info("onboarding_correction | phone=%s text=%s", phone, text)

    current_inventory = session.get("inventory", [])
    send_typing_indicator(phone, "")

    # Simple prompt to update JSON
    system_prompt = """You are an inventory assistant.
    Update the inventory list based on the user's correction.

    RULES:
    - If user sets a quantity (e.g. "Sneakers are 20"), update 'qty'.
    - If user sets a price (e.g. "Rice is 500"), update 'price'.
    - If user says remove/delete, remove the item.
    - If user adds an item, add it.
    - Match items fuzzily (e.g. "shoes" matches "Sneakers").
    - Return ONLY the updated JSON list.
    """

    prompt = f"""
    Current Inventory:
    {json.dumps(current_inventory)}

    User Correction:
    "{text}"

    Return JSON only.
    """

    try:
        claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await claude.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        updated_inventory = json.loads(raw)

        session["inventory"] = updated_inventory
        state.sessions[phone] = session

        # Re-verify with updated data (skipping full audit re-run for speed, just update receipt)
        # We pass a minimal audit result to preserve the 'estimated_total_value' logic
        # recalculate value locally
        new_value = sum(
            i.get("qty", 0) * i.get("price", 0)
            for i in updated_inventory
            if i.get("price")
        )

        # Preserve record strength but mark as manually edited
        strength = session.get("record_strength", 0)

        audit_update = {
            "estimated_total_value_ghs": new_value,
            "record_strength_score": strength,
            "verification_status": "unverified",  # Edits break the photo-match link
            "insight": "Record updated based on your feedback.",
            "risk_flag": "none",
        }

        await step_4_trigger_verification(
            phone, updated_inventory, audit_result=audit_update
        )

    except Exception:
        logger.exception("correction_error | phone=%s", phone)
        send_text(
            phone, "I couldn't make that change. Please try typing the full list again."
        )
        # Fallback to manual entry
        state.sessions[phone]["step"] = "AWAITING_TEXT_STOCK"


# ─────────────────────────────────────────────
# STEP 5 — Handle Flow submission or button confirm
# ─────────────────────────────────────────────
async def step_5_handle_flow_submission(phone: str, flow_response: dict, session: dict):
    logger.info("onboarding_step_5_start | phone=%s", phone)
    try:
        response_data = json.loads(flow_response.get("response_json", "{}"))
    except json.JSONDecodeError:
        logger.warning("onboarding_step_5_bad_json | phone=%s", phone)
        response_data = {}
    confirmed_inventory = response_data.get("inventory", session.get("inventory", []))
    session["inventory"] = confirmed_inventory
    state.sessions[phone] = session
    logger.info(
        "onboarding_step_5_inventory_confirmed | phone=%s items=%d",
        phone,
        len(confirmed_inventory),
    )
    await step_5b_ask_stock_value(phone)


async def step_5b_ask_stock_value(phone: str):
    logger.info("onboarding_step_5b_start | phone=%s", phone)
    send_text(
        phone,
        "Almost done! Two quick questions to match you with the right Shield. 🛡️\n\n"
        "1️⃣ What is the *total value* of all the stock in your shop right now?\n\n"
        "Reply with the amount in Ghana Cedis. Example: *50,000*\n\n"
        "Not sure? Type *estimate* and I will work it out from your stock.",
    )
    state.sessions[phone]["step"] = "AWAITING_CAPS_Q1"


async def step_5b_handle_stock_value(phone: str, text: str, session: dict):
    logger.info("onboarding_step_5b_response | phone=%s raw_input=%s", phone, text)
    if text.strip().lower() == "estimate":
        inventory = session.get("inventory", [])
        estimated = (
            sum(
                i.get("qty", 0) * i.get("price", 0) for i in inventory if i.get("price")
            )
            if inventory
            else 0
        )
        if estimated > 0:
            formatted = f"GHS {estimated:,.0f}"
            session["stock_value"] = float(estimated)
            state.sessions[phone] = session
            send_text(
                phone,
                f"I will use *{formatted}* as your total stock value. ✅\n\n"
                "2️⃣ If a fire or flood hit your shop tomorrow, "
                "how much money would you need to buy stock and open again?\n\n"
                "Reply with the amount in Ghana Cedis. Example: *10,000*\n\n"
                "Not sure? Type *estimate* and I will use half your stock value.",
            )
            state.sessions[phone]["step"] = "AWAITING_CAPS_Q2"
        else:
            send_text(
                phone,
                "I do not have enough price information to estimate. "
                "Please type your best guess in Ghana Cedis. Example: *50,000*",
            )
        return
    cleaned = "".join(filter(str.isdigit, text))
    if not cleaned:
        logger.warning("onboarding_step_5b_invalid_input | phone=%s", phone)
        send_text(
            phone,
            "Please reply with a number. Example: *50,000*\n\n"
            "What is the total value of all the stock in your shop?\n\n"
            "Not sure? Type *estimate*.",
        )
        return
    session["stock_value"] = float(cleaned)
    state.sessions[phone] = session
    logger.info(
        "onboarding_step_5b_stock_value_saved | phone=%s value=%s",
        phone,
        cleaned,
    )
    send_text(
        phone,
        "2️⃣ If a fire or flood hit your shop tomorrow, "
        "how much money would you need to buy stock and open again?\n\n"
        "Reply with the amount in Ghana Cedis. Example: *10,000*\n\n"
        "Not sure? Type *estimate* and I will use half your stock value.",
    )
    state.sessions[phone]["step"] = "AWAITING_CAPS_Q2"


async def step_5c_handle_restart_cap(phone: str, text: str, session: dict):
    logger.info("onboarding_step_5c_response | phone=%s raw_input=%s", phone, text)
    if text.strip().lower() == "estimate":
        stock_value = session.get("stock_value", 0)
        estimated = round(stock_value * 0.5, -3)
        if estimated <= 0:
            estimated = 5000
        formatted = f"GHS {estimated:,.0f}"
        send_text(
            phone,
            f"I will use *{formatted}* as your Restart Cap — roughly half your stock value.\n\n"
            "You can type a different amount if you prefer.",
        )
        session["restart_cap"] = float(estimated)
        state.sessions[phone] = session
        send_typing_indicator(phone, "")
        await step_6_complete_onboarding(phone, session)
        return
    cleaned = "".join(filter(str.isdigit, text))
    if not cleaned:
        logger.warning("onboarding_step_5c_invalid_input | phone=%s", phone)
        send_text(
            phone,
            "Please reply with a number. Example: *10,000*\n\n"
            "How much would you need to restock and reopen after a flood or fire?\n\n"
            "Not sure? Type *estimate*.",
        )
        return
    session["restart_cap"] = float(cleaned)
    state.sessions[phone] = session
    logger.info(
        "onboarding_step_5c_restart_cap_saved | phone=%s value=%s",
        phone,
        cleaned,
    )
    send_typing_indicator(phone, "")
    await step_6_complete_onboarding(phone, session)


# ─────────────────────────────────────────────
# STEP 6 — Tier logic + confirmation
# ─────────────────────────────────────────────
def calculate_tier(restart_cap: float) -> dict:
    # Phase 1: Single tier for all traders
    return {"tier": "Visbl Shield", "price": "GHS 50/month"}


async def step_6_complete_onboarding(phone: str, session: dict):
    logger.info("onboarding_step_6_start | phone=%s", phone)
    tier = calculate_tier(session["restart_cap"])
    session["tier"] = tier

    # Phase 1: Fixed GHS 50 (5000 pesewas)
    _TIER_PREMIUM_PESEWAS = {
        "Visbl Shield": 5000,
    }

    try:
        db = SessionLocal()
        try:
            owner = Owner(
                phone_number=phone,
                name=session.get("name"),
                shop_name=session.get("shop_name"),
                location=session.get("location"),
                category=session.get("category"),
                record_strength=session.get("record_strength", 0),
            )
            db.add(owner)
            db.flush()
            db.add(
                InventoryDeclaration(
                    owner_id=owner.id,
                    total_stock_value_ghs=session["stock_value"],
                    item_breakdown_json=json.dumps(session["inventory"]),
                )
            )
            db.add(
                Policy(
                    owner_id=owner.id,
                    status="pending",
                    premium_pesewas=_TIER_PREMIUM_PESEWAS.get(tier["tier"], 5000),
                    payout_cap_pesewas=int(session["restart_cap"] * 100),
                )
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        logger.info(
            "onboarding_step_6_db_write_success | phone=%s tier=%s",
            phone,
            tier["tier"],
        )
    except Exception:
        logger.exception("onboarding_step_6_db_write_error | phone=%s", phone)
        send_text(
            phone,
            "Something went wrong saving your record. Please try again in a moment.",
        )
        return

    restart_formatted = f"GHS {int(session['restart_cap']):,}"
    stock_formatted = f"GHS {int(session['stock_value']):,}"
    name = session.get("name", "")
    shop = session.get("shop_name", "your shop")

    send_text(
        phone,
        f"{'*' + name + '*, you' if name else 'You'} are now on Visbl! 🎉\n\n"
        f"*{shop}* has been registered with:\n"
        f"• Stock value: *{stock_formatted}*\n"
        f"• Restart Cap: *{restart_formatted}*\n"
        f"• Plan: *{tier['tier']}* — {tier['price']}\n\n"
        f"*Your record is not protected yet* - but it starts building from today.\n\n"
        f"Every day you log, your proof gets stronger. "
        f"When something goes wrong, that proof is what gets you paid. 📋",
    )
    send_reply_buttons(
        to=phone,
        body_text=(
            f"Here is how protection works:\n\n"
            f"1️⃣ Build your record daily\n"
            f"2️⃣ Pay your monthly premium\n"
            f"3️⃣ Get verified coverage commitment 🛡️\n\n"
            f"Premium: *{tier['price']}*"
        ),
        buttons=[
            {"id": "pay_now", "title": "Activate now 🔒"},
            {"id": "pay_later", "title": "Remind me later"},
        ],
        footer_text="Visbl · Cancel anytime",
    )
    state.sessions[phone]["step"] = "AWAITING_PAYMENT_DECISION"
    logger.info(
        "onboarding_complete | phone=%s tier=%s restart_cap=%s",
        phone,
        tier["tier"],
        session["restart_cap"],
    )


# ─────────────────────────────────────────────
# EXISTING USER HANDLER (post-onboarding)
# ─────────────────────────────────────────────
async def send_daily_checkin(phone: str):
    send_reply_buttons(
        to=phone,
        body_text=(
            "Good morning! ☀️\n\nHow is your shop today? Any changes to your stock?"
        ),
        buttons=[
            {"id": "checkin_good", "title": "All good today ✅"},
            {"id": "checkin_update", "title": "Update my stock 📦"},
            {"id": "checkin_problem", "title": "Had a problem ⚠️"},
        ],
        footer_text="Visbl · Your daily record",
    )
    state.sessions[phone] = state.sessions.get(phone, {})
    state.sessions[phone]["step"] = "DAILY_CHECKIN"


async def handle_existing_user(phone: str, message: dict, owner):
    logger.info("existing_user_message | phone=%s", phone)
    session = state.sessions.get(phone, {})
    step = session.get("step", "IDLE")
    msg_type = message.get("type", "")
    text = message.get("text", {}).get("body", "").strip()
    text_lower = text.lower()
    message_id = message.get("id", "")
    interactive = message.get("interactive", {})
    button_id = interactive.get("button_reply", {}).get("id", "") or interactive.get(
        "list_reply", {}
    ).get("id", "")
    name = (owner.name if owner and owner.name else "").strip()
    greeting = f"Hi {name}!" if name else "Hey!"

    if text_lower == "cancel" and step in _CANCELLABLE_STEPS:
        logger.info("user_cancelled | phone=%s step=%s", phone, step)
        session["step"] = "IDLE"
        session.pop("daily_inventory", None)
        state.sessions[phone] = session
        _send_main_menu(phone, "Cancelled. ✅")
        return

    if text_lower in ("delete my data", "reset", "start over"):
        send_reply_buttons(
            to=phone,
            body_text=(
                "⚠️ This will permanently delete your shop record and all your data.\n\n"
                "Are you sure? This cannot be undone."
            ),
            buttons=[
                {"id": "confirm_delete", "title": "Yes, delete it"},
                {"id": "cancel_delete", "title": "No, keep my data"},
            ],
        )
        session["step"] = "AWAITING_DELETE_CONFIRM"
        state.sessions[phone] = session
        return

    if step == "AWAITING_DELETE_CONFIRM":
        if button_id == "confirm_delete":
            await _delete_user_data(phone, owner)
        elif button_id == "cancel_delete":
            _send_main_menu(phone, "Your data is safe. 👍")
            session["step"] = "IDLE"
            state.sessions[phone] = session
        else:
            send_reply_buttons(
                to=phone,
                body_text="Please tap one of the buttons to confirm.",
                buttons=[
                    {"id": "confirm_delete", "title": "Yes, delete it"},
                    {"id": "cancel_delete", "title": "No, keep my data"},
                ],
            )
        return

    if step == "AWAITING_PAYMENT_DECISION":
        if button_id == "pay_now":
            _send_payment_instructions(phone)
            session["step"] = "AWAITING_PAYMENT_CONFIRM"
            state.sessions[phone] = session
        elif button_id == "pay_later":
            send_text(
                phone,
                "No problem. Your record keeps building every day. 📋\n\n"
                "Type *ACTIVATE* anytime when you are ready.",
            )
            session["step"] = "IDLE"
            state.sessions[phone] = session
        else:
            tier = session.get("tier", {})
            price = tier.get("price", "") if isinstance(tier, dict) else ""
            send_reply_buttons(
                to=phone,
                body_text=(
                    f"Ready to activate your Shield? 🛡️\n\nFirst premium: *{price}*"
                    if price
                    else "Ready to activate your Shield? 🛡️"
                ),
                buttons=[
                    {"id": "pay_now", "title": "Activate now 🔒"},
                    {"id": "pay_later", "title": "Remind me later"},
                ],
                footer_text="Visbl · Cancel anytime",
            )
        return

    if step == "AWAITING_PAYMENT_CONFIRM":
        if text_lower == "paid":
            send_text(
                phone,
                "Thank you! 🎉 Checking your payment now.\n\n"
                "Your Shield will be active within a few minutes.",
            )
            session["step"] = "IDLE"
            state.sessions[phone] = session
        elif text_lower == "cancel":
            send_text(phone, "Cancelled. Type *ACTIVATE* anytime when you are ready.")
            session["step"] = "IDLE"
            state.sessions[phone] = session
        else:
            send_text(
                phone,
                "Reply *PAID* once you have sent the premium.\n\n"
                "Type *CANCEL* to go back.",
            )
        return

    if step == "DAILY_CHECKIN":
        if button_id == "checkin_good":
            send_text(phone, "Logged ✅ Keep it up!")
            session["step"] = "IDLE"
            state.sessions[phone] = session
        elif button_id == "checkin_update":
            await _daily_ask_input_type(phone)
        elif button_id == "checkin_problem":
            send_text(
                phone,
                "What happened? Describe it briefly and I will log it.\n\n"
                "For fire or flood, type *CLAIM*.",
            )
            session["step"] = "AWAITING_INCIDENT_REPORT"
            state.sessions[phone] = session
        else:
            await send_daily_checkin(phone)
        return

    if step in ("DAILY_AWAITING_INPUT_TYPE", "AWAITING_STOCK_UPDATE"):
        send_typing_indicator(phone, message_id)
        if not button_id:
            await _daily_ask_input_type(phone)
            return
        await _daily_handle_input_choice(phone, button_id)
        return

    if step == "DAILY_AWAITING_TEXT":
        if msg_type == "text" and text:
            await _daily_handle_text(phone, text, session)
        elif msg_type == "audio":
            await _daily_handle_voice(phone, message, session)
        else:
            send_reply_buttons(
                to=phone,
                body_text=(
                    "Please type your stock list or send a voice note. 🎤\n\n"
                    "Example: Sneakers: 15, Heels: 10, Bags: 5"
                ),
                buttons=[
                    {"id": "daily_voice", "title": "🎤 Try voice again"},
                    {"id": "daily_text", "title": "✍️ Type instead"},
                ],
                footer_text="Visbl · Type *cancel* to stop",
            )
        return

    if step == "DAILY_AWAITING_PHOTO":
        if msg_type == "image":
            await _daily_handle_photo(phone, message, session)
        else:
            # Race condition & Buffering check
            if state.sessions.get(phone, {}).get("image_buffer"):
                logger.info("buffering_skip | phone=%s", phone)
                return

            await asyncio.sleep(2.0)

            # Check buffer again
            if state.sessions.get(phone, {}).get("image_buffer"):
                logger.info("buffering_skip_after_sleep | phone=%s", phone)
                return

            current_step = state.sessions.get(phone, {}).get("step")
            if current_step == "DAILY_PROCESSING_PHOTO":
                logger.info(
                    "race_condition_skip | phone=%s step=%s", phone, current_step
                )

                return

            send_text(
                phone,
                "Please send a photo of your shelves or log book. 📸\n\n"
                "Type *cancel* to go back.",
            )
        return

    if step == "DAILY_AWAITING_CONFIRM":
        if button_id == "daily_confirm":
            inventory = session.get("daily_inventory", [])
            await _daily_save_inventory(phone, owner, inventory)
        elif button_id == "daily_edit":
            send_text(
                phone,
                "No problem — type your corrected list below.\n\n"
                "Example: Sneakers: 20, Heels: 5, Bags: 3\n\n"
                "Type *cancel* to go back.",
            )
            session["step"] = "DAILY_AWAITING_TEXT"
            state.sessions[phone] = session
        else:
            inventory = session.get("daily_inventory", [])
            await _daily_confirm_inventory(phone, inventory)
        return

    if button_id == "menu_log":
        await _daily_ask_input_type(phone)
        return
    if button_id == "menu_report":
        send_text(
            phone,
            "📄 *Your Shop Record — GHS 2*\n\n"
            "Send *GHS 2* via Mobile Money:\n"
            "*0XX XXX XXXX* (Visbl) · Reference: REPORT\n\n"
            "Reply *PAID* once done, or *CANCEL* to go back.",
        )
        session["step"] = "AWAITING_REPORT_PAYMENT"
        state.sessions[phone] = session
        return
    if button_id == "menu_activate":
        send_reply_buttons(
            to=phone,
            body_text="Ready to activate your Shield? 🛡️\n\nYour first premium locks in your protection.",
            buttons=[
                {"id": "pay_now", "title": "Activate now 🔒"},
                {"id": "pay_later", "title": "Not yet"},
            ],
        )
        session["step"] = "AWAITING_PAYMENT_DECISION"
        state.sessions[phone] = session
        return
    if button_id == "menu_delete":
        send_reply_buttons(
            to=phone,
            body_text=(
                "⚠️ This will permanently delete your shop record and all your data.\n\n"
                "Are you sure? This cannot be undone."
            ),
            buttons=[
                {"id": "confirm_delete", "title": "Yes, delete it"},
                {"id": "cancel_delete", "title": "No, keep my data"},
            ],
        )
        session["step"] = "AWAITING_DELETE_CONFIRM"
        state.sessions[phone] = session
        return

    if text_lower in ("log", "update", "update stock"):
        await _daily_ask_input_type(phone)
        return
    if text_lower == "activate":
        send_reply_buttons(
            to=phone,
            body_text="Ready to activate your Shield? 🛡️\n\nYour first premium locks in your protection.",
            buttons=[
                {"id": "pay_now", "title": "Activate now 🔒"},
                {"id": "pay_later", "title": "Not yet"},
            ],
        )
        session["step"] = "AWAITING_PAYMENT_DECISION"
        state.sessions[phone] = session
        return
    if button_id == "pay_now":
        _send_payment_instructions(phone)
        session["step"] = "AWAITING_PAYMENT_CONFIRM"
        state.sessions[phone] = session
        return
    if button_id == "pay_later":
        send_text(phone, "Got it. Type *ACTIVATE* anytime when you are ready. 👍")
        session["step"] = "IDLE"
        state.sessions[phone] = session
        return
    if text_lower == "report":
        send_text(
            phone,
            "📄 *Your Shop Record — GHS 2*\n\n"
            "Send *GHS 2* via Mobile Money:\n"
            "*0XX XXX XXXX* (Visbl) · Reference: REPORT\n\n"
            "Reply *PAID* once done, or *CANCEL* to go back.",
        )
        session["step"] = "AWAITING_REPORT_PAYMENT"
        state.sessions[phone] = session
        return

    if step == "AWAITING_REPORT_PAYMENT":
        if text_lower == "paid":
            send_typing_indicator(phone, message_id)
            send_text(phone, "Checking payment... generating your report now. 📄")
            await _send_shop_report(phone, owner)
            session["step"] = "IDLE"
            state.sessions[phone] = session
        elif text_lower == "cancel":
            send_text(phone, "Cancelled. Type *REPORT* anytime when you are ready.")
            session["step"] = "IDLE"
            state.sessions[phone] = session
        else:
            send_text(
                phone,
                "Send *GHS 2* to *0XX XXX XXXX* (Visbl) · Reference: REPORT\n\n"
                "Reply *PAID* once done, or *CANCEL* to go back.",
            )
        return

    if step in ("DAILY_PROCESSING_PHOTO", "PROCESSING_PHOTO"):
        logger.info("concurrent_webhook_skip | phone=%s step=%s", phone, step)
        return

    _send_main_menu(phone, f"{greeting} Good to hear from you. 👋")


async def _send_shop_report(phone: str, owner):
    logger.info("report_generate_start | phone=%s", phone)
    if not owner:
        send_text(phone, "I could not find your shop record. Please try again.")
        return
    send_text(phone, "Generating your shop record... 📄")
    send_typing_indicator(phone, "")
    db = SessionLocal()
    try:
        owner_id = int(owner.id)
        fresh_owner = db.query(Owner).filter(Owner.id == owner_id).first()
        declaration = (
            db.query(InventoryDeclaration)
            .filter(InventoryDeclaration.owner_id == owner_id)
            .order_by(InventoryDeclaration.generated_at.desc())
            .first()
        )
        policy = (
            db.query(Policy)
            .filter(Policy.owner_id == owner_id)
            .order_by(Policy.id.desc())
            .first()
        )
        owner_data = {
            "name": fresh_owner.name if fresh_owner else None,
            "shop_name": fresh_owner.shop_name if fresh_owner else None,
            "location": fresh_owner.location if fresh_owner else None,
            "category": fresh_owner.category if fresh_owner else None,
            "phone_number": phone,
            "created_at": fresh_owner.created_at if fresh_owner else None,
        }
        declaration_data = (
            {
                "total_stock_value_ghs": float(declaration.total_stock_value_ghs or 0),
                "item_breakdown_json": declaration.item_breakdown_json,
                "generated_at": declaration.generated_at,
            }
            if declaration
            else None
        )
        policy_data = (
            {
                "status": policy.status,
                "premium_pesewas": policy.premium_pesewas,
                "payout_cap_pesewas": policy.payout_cap_pesewas,
                "cover_start_date": policy.cover_start_date,
                "last_premium_paid_at": policy.last_premium_paid_at,
            }
            if policy
            else None
        )
    except Exception as e:
        logger.exception("report_db_fetch_error | phone=%s error=%s", phone, e)
        send_text(
            phone,
            "Something went wrong generating your report. Please try again.",
        )
        return
    finally:
        db.close()
    try:
        pdf_bytes = await asyncio.to_thread(
            generate_report_pdf, owner_data, declaration_data, policy_data
        )
    except Exception as e:
        logger.exception("report_pdf_build_error | phone=%s error=%s", phone, e)
        send_text(
            phone,
            "Something went wrong generating your report. Please try again.",
        )
        return
    media_id = await asyncio.to_thread(upload_pdf_to_whatsapp, pdf_bytes)
    if not media_id:
        logger.error("report_upload_failed | phone=%s", phone)
        send_text(
            phone,
            "I could not send the report right now. Please try again later.",
        )
        return
    shop_name = (owner_data.get("shop_name") or "Shop").replace(" ", "_")
    filename = f"Visbl_{shop_name}_Record.pdf"
    send_document(
        to=phone,
        media_id=media_id,
        filename=filename,
        caption=(
            "Here is your Visbl shop record. 📋\n\n"
            "Keep this document safe - it is your proof of stock."
        ),
    )
    logger.info("report_sent | phone=%s filename=%s", phone, filename)


# ─────────────────────────────────────────────
# DAILY UPDATE HELPERS
# ─────────────────────────────────────────────
async def _daily_ask_input_type(phone: str):
    send_list_message(
        to=phone,
        body_text="How would you like to update your stock today? 📦",
        button_label="Choose method",
        sections=[
            {
                "title": "Pick one",
                "rows": [
                    {
                        "id": "daily_photo",
                        "title": "📸 Send a photo",
                        "description": "Take a picture of your shelves",
                    },
                    {
                        "id": "daily_logbook",
                        "title": "📖 Log book page",
                        "description": "Photo of your written records",
                    },
                    {
                        "id": "daily_voice",
                        "title": "🎤 Voice note",
                        "description": "Speak your stock list",
                    },
                    {
                        "id": "daily_text",
                        "title": "✍️ Type it out",
                        "description": "Type your list as a message",
                    },
                ],
            }
        ],
        footer_text="Visbl · Daily record · Type *cancel* to stop",
    )
    session = state.sessions.get(phone, {})
    session["step"] = "DAILY_AWAITING_INPUT_TYPE"
    state.sessions[phone] = session


async def _daily_handle_input_choice(phone: str, button_id: str):
    session = state.sessions.get(phone, {})
    if button_id in ("daily_photo", "daily_logbook"):
        send_text(
            phone,
            "Take a clear photo of your shelves or log book page and send it here. 📸\n\n"
            "Make sure everything is in the frame and well lit.\n\n"
            "Type *cancel* to go back.",
        )
        session["step"] = "DAILY_AWAITING_PHOTO"
        state.sessions[phone] = session
    elif button_id == "daily_voice":
        send_text(
            phone,
            "Send me a voice note now. 🎤\n\n"
            "Say something like:\n"
            '_"Sneakers 15, Heels 10 at 120 cedis, Bags 5"_\n\n'
            "I will transcribe it and count your stock.\n\n"
            "Type *cancel* to go back.",
        )
        session["step"] = "DAILY_AWAITING_TEXT"
        state.sessions[phone] = session
    elif button_id == "daily_text":
        send_text(
            phone,
            "Type your stock list below.\n\n"
            "Example:\n"
            "Sneakers: 15 @ GHS 120\n"
            "Heels: 10\n"
            "Bags: 5 @ GHS 80\n\n"
            "Add *@ GHS price* after a quantity if you know the unit price.\n\n"
            "Type *cancel* to go back.",
        )
        session["step"] = "DAILY_AWAITING_TEXT"
        state.sessions[phone] = session
    else:
        session["step"] = "IDLE"
        state.sessions[phone] = session
        await _daily_ask_input_type(phone)


async def _daily_handle_voice(phone: str, message: dict, session: dict):
    t0 = time.perf_counter()
    logger.info("daily_voice_start | phone=%s", phone)
    message_id = message["id"]
    media_id = message.get("audio", {}).get("id")
    if not media_id:
        send_reply_buttons(
            to=phone,
            body_text="I could not read that voice note. Please try again. 🎤",
            buttons=[
                {"id": "daily_voice", "title": "🎤 Try again"},
                {"id": "daily_text", "title": "✍️ Type instead"},
            ],
            footer_text="Visbl · Type *cancel* to stop",
        )
        session["step"] = "DAILY_AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url_resp = await client.get(
                f"https://graph.facebook.com/v22.0/{media_id}",
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            )
        media_url = url_resp.json().get("url")
    except Exception:
        logger.exception("daily_voice_url_error | phone=%s", phone)
        send_reply_buttons(
            to=phone,
            body_text="I could not reach WhatsApp. Please try again. 🎤",
            buttons=[
                {"id": "daily_voice", "title": "🎤 Try again"},
                {"id": "daily_text", "title": "✍️ Type instead"},
            ],
            footer_text="Visbl · Type *cancel* to stop",
        )
        session["step"] = "DAILY_AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    if not media_url:
        send_reply_buttons(
            to=phone,
            body_text="I could not read that voice note. Please try again. 🎤",
            buttons=[
                {"id": "daily_voice", "title": "🎤 Try again"},
                {"id": "daily_text", "title": "✍️ Type instead"},
            ],
            footer_text="Visbl · Type *cancel* to stop",
        )
        session["step"] = "DAILY_AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            audio_resp, _ = await asyncio.gather(
                client.get(
                    media_url,
                    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                ),
                asyncio.to_thread(
                    send_text,
                    phone,
                    "Got it! Listening to your voice note... 🎤⏳",
                ),
            )
    except Exception:
        logger.exception("daily_voice_download_error | phone=%s", phone)
        send_reply_buttons(
            to=phone,
            body_text="I had trouble downloading your voice note. Please try again. 🎤",
            buttons=[
                {"id": "daily_voice", "title": "🎤 Try again"},
                {"id": "daily_text", "title": "✍️ Type instead"},
            ],
            footer_text="Visbl · Type *cancel* to stop",
        )
        session["step"] = "DAILY_AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    send_typing_indicator(phone, message_id)
    transcript = await _transcribe_audio(phone, audio_resp.content)
    if not transcript:
        send_reply_buttons(
            to=phone,
            body_text=(
                "I could not understand that voice note. 🎤\n\n"
                'Speak clearly: _"Sneakers 15, Heels 10, Bags 5"_'
            ),
            buttons=[
                {"id": "daily_voice", "title": "🎤 Try again"},
                {"id": "daily_text", "title": "✍️ Type instead"},
            ],
            footer_text="Visbl · Type *cancel* to stop",
        )
        session["step"] = "DAILY_AWAITING_INPUT_TYPE"
        state.sessions[phone] = session
        return
    logger.info(
        "daily_voice_transcribed | phone=%s transcript=%s elapsed=%.2fs",
        phone,
        transcript,
        time.perf_counter() - t0,
    )
    await _daily_handle_text(phone, transcript, session)


async def _transcribe_audio(phone: str, audio_bytes: bytes) -> str | None:
    t0 = time.perf_counter()
    logger.info(
        "transcribe_start | phone=%s size_kb=%.1f",
        phone,
        len(audio_bytes) / 1024,
    )
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    providers = []
    if groq_key:
        providers.append(
            (
                "groq",
                "https://api.groq.com/openai/v1/audio/transcriptions",
                groq_key,
                "whisper-large-v3",
            )
        )
    if openai_key:
        providers.append(
            (
                "openai",
                "https://api.openai.com/v1/audio/transcriptions",
                openai_key,
                "whisper-1",
            )
        )
    if not providers:
        logger.error(
            "transcribe_no_key | phone=%s reason=no_groq_or_openai_key",
            phone,
        )
        return None
    for provider, url, api_key, model in providers:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={"model": model},
                    files={"file": ("audio.ogg", audio_bytes, "audio/ogg")},
                )
                response.raise_for_status()
                transcript = response.json().get("text", "").strip()
            logger.info(
                "transcribe_complete | phone=%s provider=%s elapsed=%.2fs",
                phone,
                provider,
                time.perf_counter() - t0,
            )
            return transcript or None
        except Exception:
            logger.exception("transcribe_error | phone=%s provider=%s", phone, provider)
    return None


async def _daily_handle_text(phone: str, text: str, session: dict):
    inventory = await parse_text_inventory(text)
    if not inventory:
        send_text(
            phone,
            "I couldn't read that list. Please use this format:\n\n"
            "Sneakers: 15\nHeels: 10\nBags: 5\n\n"
            "Or: Sneakers 15, Heels 10, Bags 5\n\n"
            "Type *cancel* to go back.",
        )
        return
    session["daily_inventory"] = inventory
    state.sessions[phone] = session
    await _daily_confirm_inventory(phone, inventory)


async def _daily_handle_photo(phone: str, message: dict, session: dict):
    # ── BUFFERING LOGIC ──
    logger.info("daily_photo_buffer | phone=%s", phone)
    live = state.sessions.get(phone, {})

    if live.get("step") == "DAILY_PROCESSING_PHOTO":
        return

    if "image_buffer" not in live:
        live["image_buffer"] = []

    live["image_buffer"].append(message)

    deadline = time.time() + 2.0
    live["buffer_deadline"] = deadline
    state.sessions[phone] = live

    asyncio.create_task(_daily_buffer_waiter(phone, deadline))


async def _daily_buffer_waiter(phone, task_deadline):
    await asyncio.sleep(2.5)

    live = state.sessions.get(phone, {})
    current_deadline = live.get("buffer_deadline", 0)

    if current_deadline > task_deadline:
        return

    messages = live.get("image_buffer", [])
    if not messages:
        return

    live["step"] = "DAILY_PROCESSING_PHOTO"
    live["image_buffer"] = []
    live.pop("buffer_deadline", None)
    state.sessions[phone] = live

    await _daily_process_images(phone, messages)


async def _daily_process_images(phone: str, messages: list):
    t0 = time.perf_counter()
    logger.info("daily_photo_start | phone=%s count=%d", phone, len(messages))

    last_message = messages[-1]
    message_id = last_message["id"]

    send_text(phone, "Got it! Counting your stock now... ⏳")
    send_typing_indicator(phone, message_id)

    image_b64s = []

    for message in messages:
        media_ids = (
            [message["image"]]
            if isinstance(message["image"], dict)
            else message["image"]
        )
        for media_id_obj in media_ids:
            media_id = media_id_obj["id"]
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    url_resp = await client.get(
                        f"https://graph.facebook.com/v22.0/{media_id}",
                        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                    )
                media_url = url_resp.json().get("url")
            except Exception:
                logger.exception("daily_photo_url_error | phone=%s", phone)
                continue

            if not media_url:
                continue

            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    img_resp = await client.get(
                        media_url,
                        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                    )
                image_b64s.append(base64.b64encode(img_resp.content).decode("utf-8"))
            except Exception:
                logger.exception("daily_photo_download_error | phone=%s", phone)
                continue

    if not image_b64s:
        state.sessions[phone]["step"] = "DAILY_AWAITING_PHOTO"
        send_text(
            phone,
            "I could not download your photos. Please try again. 📸",
        )
        return

    # ── CHANGED: pull context before calling Claude ──
    previous_logs, record_strength, restart_cap = _get_owner_context(phone)

    result = await step_4_parse_inventory_with_claude(
        phone,
        image_b64s,
        previous_logs=previous_logs,
        current_record_strength=record_strength,
        restart_cap=restart_cap,
    )

    inventory = result.get("inventory", [])
    if not inventory:
        state.sessions[phone]["step"] = "DAILY_AWAITING_PHOTO"
        send_text(
            phone,
            "I could not read the stock clearly from that photo. "
            "Please try again with a clearer, well-lit picture. 📸\n\n"
            "Type *cancel* to go back.",
        )
        return

    logger.info(
        "daily_photo_complete | phone=%s items=%d elapsed=%.2fs",
        phone,
        len(inventory),
        time.perf_counter() - t0,
    )

    session = state.sessions.get(phone, {})
    session["daily_inventory"] = inventory
    session["record_strength"] = result.get("record_strength_score", 0)
    state.sessions[phone] = session

    # ── CHANGED: use receipt format from audit result ──
    await _daily_confirm_inventory(phone, inventory, audit_result=result)


async def _daily_confirm_inventory(
    phone: str,
    inventory: list,
    audit_result: dict = None,
):
    def _fmt(i: dict) -> str:
        line = f"• {i['item']}: {i['qty']} pieces"
        if i.get("price"):
            line += f" @ GHS {i['price']:,.0f}"
        if i.get("date"):
            line += f" (logged {i['date']})"
        return line

    items_text = "\n".join(_fmt(i) for i in inventory)

    # Truncate if too long for WhatsApp Interactive Message (Limit is ~1024 chars)
    MAX_BODY_CHARS = 1000
    if len(items_text) > MAX_BODY_CHARS:
        truncated = items_text[:MAX_BODY_CHARS]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        remaining_count = len(inventory) - truncated.count("\n") - 1
        items_text = truncated + f"\n... and {remaining_count} more items"

    # Build receipt if we have audit data

    if audit_result:
        total_value = audit_result.get("estimated_total_value_ghs", 0)
        strength = audit_result.get("record_strength_score", 0)
        insight = audit_result.get("insight", "")
        verification_status = audit_result.get("verification_status", "unverified")

        if verification_status == "match":
            badge = "✓ Matches your records"
        elif verification_status == "mismatch":
            badge = "⚠️ Something looks different"
        else:
            badge = "📋 Recorded by Visbl AI"

        body = (
            f"Visbl BUSINESS RECEIPT 🧾\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Date: {datetime.now().strftime('%d %b %Y')}\n\n"
            f"Stock recorded:\n{items_text}\n\n"
            f"Est. Total Value: GHS {total_value:,}\n"
            f"{badge}\n\n"
            # f"Record Strength: {strength}/100\n"
            f"━━━━━━━━━━━━━━━━━━"
        )
        if insight:
            body += f"\n💡 {insight}"
        if verification_status == "mismatch":
            body += (
                "\n\n⚠️ Stock looks lower than your last record. "
                "If you sold items, reply with how many so I can update."
            )
        body += "\n\nDoes this look correct?"
    else:
        body = f"Here is what I counted:\n\n{items_text}\n\nDoes this look correct?"

    send_reply_buttons(
        to=phone,
        header_text="Today's Stock Update",
        body_text=body,
        buttons=[
            {"id": "daily_confirm", "title": "Yes, save it ✅"},
            {"id": "daily_edit", "title": "No, edit it"},
        ],
        footer_text="Visbl · Daily record · Type *cancel* to stop",
    )
    session = state.sessions.get(phone, {})
    session["step"] = "DAILY_AWAITING_CONFIRM"
    state.sessions[phone] = session


async def _daily_save_inventory(phone: str, owner, inventory: list):
    logger.info("daily_save_start | phone=%s items=%d", phone, len(inventory))
    owner_id = int(owner.id) if owner else None
    if not owner_id:
        send_text(phone, "Something went wrong saving your update. Please try again.")
        return
    total_value = sum(
        i.get("qty", 0) * i.get("price", 0) for i in inventory if i.get("price")
    )
    send_typing_indicator(phone, "")
    db = SessionLocal()
    db_error = False
    try:
        db.add(
            InventoryDeclaration(
                owner_id=owner_id,
                total_stock_value_ghs=total_value if total_value > 0 else None,
                item_breakdown_json=json.dumps(inventory),
            )
        )
        for item in inventory:
            price_pesewas = int(item["price"] * 100) if item.get("price") else None
            stock_value_pesewas = (
                int(item.get("qty", 0) * item["price"] * 100)
                if item.get("price")
                else None
            )
            db.add(
                InventoryLog(
                    owner_id=owner_id,
                    entry_type="daily_snapshot",
                    product_name=item.get("item"),
                    quantity=item.get("qty"),
                    unit_price_pesewas=price_pesewas,
                    stock_value_pesewas=stock_value_pesewas,
                    raw_message=json.dumps(item),
                )
            )

        # Update owner record strength if changed in session
        session = state.sessions.get(phone, {})
        new_strength = session.get("record_strength")
        if new_strength is not None:
            owner_rec = db.query(Owner).filter(Owner.id == owner_id).first()
            if owner_rec:
                owner_rec.record_strength = new_strength

        db.commit()
        logger.info("daily_save_complete | phone=%s", phone)
    except Exception as e:
        db.rollback()
        db_error = True
        logger.exception("daily_save_error | phone=%s error=%s", phone, e)
    finally:
        db.close()

    session = state.sessions.get(phone, {})
    session["step"] = "IDLE"
    session.pop("daily_inventory", None)
    state.sessions[phone] = session

    if db_error:
        send_text(phone, "Something went wrong saving your update. Please try again.")
    else:
        send_text(
            phone,
            "Stock saved! ✅\n\n"
            "Your record is building - see you tomorrow. 📋\n\n"
            "*LOG* anytime to update again.",
        )


def _send_payment_instructions(phone: str):
    send_text(
        phone,
        "Here is how to pay your first premium:\n\n"
        "*Mobile Money:* Send to *0XX XXX XXXX* (Visbl)\n"
        "Reference: your phone number\n\n"
        "Reply *PAID* once done.",
    )


async def _delete_user_data(phone: str, owner):
    logger.info("user_data_delete_start | phone=%s", phone)
    owner_id = int(owner.id) if owner else None
    db_error = False
    db = SessionLocal()
    try:
        if owner_id:
            policy_ids = [
                r[0]
                for r in db.query(Policy.id).filter(Policy.owner_id == owner_id).all()
            ]
            if policy_ids:
                db.query(Claim).filter(Claim.policy_id.in_(policy_ids)).delete(
                    synchronize_session=False
                )
            db.query(Policy).filter(Policy.owner_id == owner_id).delete(
                synchronize_session=False
            )
            db.query(InventoryLog).filter(InventoryLog.owner_id == owner_id).delete(
                synchronize_session=False
            )
            db.query(InventoryDeclaration).filter(
                InventoryDeclaration.owner_id == owner_id
            ).delete(synchronize_session=False)
            db.query(Owner).filter(Owner.id == owner_id).delete(
                synchronize_session=False
            )
        db.commit()
        logger.info("user_data_delete_complete | phone=%s", phone)
    except Exception:
        db.rollback()
        db_error = True
        logger.exception("user_data_delete_error | phone=%s", phone)
    finally:
        db.close()
    state.sessions.pop(phone, None)
    if db_error:
        send_text(phone, "Something went wrong deleting your data. Please try again.")
    else:
        send_text(
            phone,
            "Done. Your account has been deleted. 🗑️\n\nSend *Hi* to start fresh.",
        )


def _send_main_menu(phone: str, greeting: str = ""):
    body = (f"{greeting}\n\n" if greeting else "") + "What would you like to do?"
    send_list_message(
        to=phone,
        body_text=body,
        button_label="Choose an option",
        sections=[
            {
                "title": "Your shop",
                "rows": [
                    {
                        "id": "menu_log",
                        "title": "Update stock 📦",
                        "description": "Log today's stock",
                    },
                    {
                        "id": "menu_report",
                        "title": "My record 📄",
                        "description": "Download your shop record",
                    },
                    {
                        "id": "menu_activate",
                        "title": "Activate Shield 🛡️",
                        "description": "Start your protection",
                    },
                    {
                        "id": "menu_delete",
                        "title": "Delete my account 🗑️",
                        "description": "Remove all your data",
                    },
                ],
            }
        ],
        footer_text="Visbl · Your shop record",
    )


def parse_inventory(raw: str):
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        return None
    except Exception as e:
        logger.error(
            "parse_inventory_json_error | error=%s raw=%r",
            e,
            raw[:200] if raw else None,
        )
        return None


async def parse_text_inventory(text: str) -> list:
    """
    Use LLM to extract structured inventory from natural language text.
    Falls back to regex if LLM fails.
    """
    logger.info("parse_text_inventory_start | text=%r", text[:50])

    # 1. Try Regex first for simple structured lists (faster/cheaper)
    # If it looks like a list "Item: 5", regex is fine.
    # But for "I bought 5 shoes", regex might fail or be less robust.
    # Actually, let's try LLM first for robustness with voice notes,
    # unless it's a very clear list format.

    # Simple heuristic: if it contains newlines and colons, it might be a typed list
    if "\n" in text and ":" in text:
        regex_result = _parse_text_inventory_regex(text)
        if regex_result:
            return regex_result

    # 2. Use Claude Haiku for natural language parsing
    system_prompt = """You are a stock inventory parser for a market trader.
    Extract items, quantities, and prices from the text.

    RULES:
    - Return a JSON list of objects: [{"item": "string", "qty": int, "price": float (optional)}]
    - If quantity is missing or vague ("some"), default to 1.
    - If price is mentioned ("at 50", "50 cedis"), include it.
    - Handle "5 pairs of jeans" -> item: "Jeans", qty: 5.
    - Handle "5 shirts at 20" -> price is 20 (unit price).
    - Handle "5 shirts for 100" -> price is 20 (100/5) if clear, else 100.
    - Ignore conversational filler ("I bought", "and then", "please log").
    - Return ONLY raw JSON. No markdown.
    """

    try:
        claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await claude.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": text}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)

        if isinstance(data, list) and data:
            logger.info("parse_text_llm_success | items=%d", len(data))
            return data

    except Exception:
        logger.exception("parse_text_llm_failed | falling_back_to_regex")

    # 3. Fallback to regex
    return _parse_text_inventory_regex(text)


def _parse_text_inventory_regex(text: str) -> list:
    """Legacy regex parser for structured lists."""
    items = []
    # Matches "5 Shoes" or "Shoes: 5"
    pattern = re.compile(
        r"(?:(\d+)\s+([A-Za-z][A-Za-z\s&/'\-]+?))"
        r"|(?:([A-Za-z][A-Za-z\s&/'\-]+?)\s*[:\-]?\s*(\d+))",
        re.IGNORECASE,
    )
    # Matches price "@ 50" or "at 50"
    price_pattern = re.compile(
        r"(?:@|at|for)\s*(?:GHS|cedis)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
    )

    for line in re.split(r"[,\n]+", text):
        line = line.strip()
        if not line:
            continue

        m = pattern.search(line)
        if not m:
            continue

        if m.group(1) and m.group(2):
            qty, name = int(m.group(1)), m.group(2).strip().rstrip(",").strip()
        else:
            name, qty = (
                m.group(3).strip().rstrip(",").strip(),
                int(m.group(4)),
            )

        if not name:
            continue

        entry: dict = {"item": name.title(), "qty": qty}

        # Try to find price
        pm = price_pattern.search(line)
        if pm:
            entry["price"] = float(pm.group(1))

        items.append(entry)

    return items


def _repair_json(raw: str) -> str:
    """
    Fix common JSON errors Claude produces when transcribing handwritten records:
    1. Unquoted keys:  qty: 1  →  "qty": 1
    2. Trailing commas before } or ]
    3. Single quotes instead of double quotes
    """
    # Fix unquoted keys — word characters before a colon with no opening quote
    raw = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', raw)

    # Fix trailing commas before closing braces/brackets
    raw = re.sub(r",\s*([\]}])", r"\1", raw)

    # Fix single-quoted strings → double-quoted
    raw = re.sub(r"'([^']*)'", r'"\1"', raw)

    return raw
