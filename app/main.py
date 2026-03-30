import asyncio
import logging
import os

from dotenv.main import load_dotenv
from fastapi import Depends, FastAPI, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sqlalchemy.orm import Session

import state
from app.DB.database import SessionLocal, engine
from app.DB.models import Base, Owner
from app.handlers import onboarding
from app.handlers.flow_exchange import router as flow_router
from app.handlers.onboarding import (
    step_1b_ask_name,
    step_1c_handle_name,
    step_1d_handle_shop,
    step_1e_handle_location,
    step_1f_handle_category,
    step_2b_handle_input_choice,
    step_2c_handle_text_stock,
    step_3_handle_photo,
    step_3_handle_voice,
    step_5_handle_flow_submission,
    step_5b_handle_stock_value,
    step_5c_handle_restart_cap,
)
from app.handlers.voice_transcriber import transcribe_voice_message
from app.handlers.whatsapp_manager import (
    send_reply_buttons,
    send_text,
    send_typing_indicator,
)
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.include_router(flow_router)

load_dotenv()

WHATSAPP_SECRET_KEY = os.getenv("WHATSAPP_SECRET_KEY")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    img = Image.open("assets/greeting_out.png")
    img = img.convert("RGB")
    img = img.resize((800, int(img.height * 800 / img.width)))
    img.save("assets/greetings.jpg", "JPEG", quality=60, optimize=True)
    return {"message": "Visbl is active!"}


@app.get("/webhook")
async def verify(req: Request):
    params = req.query_params
    verify_token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if verify_token == WHATSAPP_SECRET_KEY:
        return Response(content=challenge, status_code=200)

    return Response(content="Forbidden", status_code=403)


@app.post("/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    phone = None  # keep in scope so the except block can notify the user
    try:
        data = await request.json()

        if "object" not in data:
            logger.warning("webhook_missing_object_key")
            return Response(status_code=400)

        logger.debug("whatsapp_webhook_payload | data=%s", data)

        value = data["entry"][0]["changes"][0]["value"]

        # ── Status update (delivered, read, failed, sent) ──
        if "statuses" in value and "messages" not in value:
            _log_message_status(value["statuses"][0])
            return Response(status_code=200)

        # ── No messages and no statuses — unknown payload, skip ──
        if "messages" not in value:
            logger.warning("webhook_unknown_payload | keys=%s", list(value.keys()))
            return Response(status_code=200)

        # ── Incoming message ──
        message = value["messages"][0]
        phone = message["from"]
        msg_type = message["type"]
        message_id = message["id"]

        # Try to get user name from contacts
        user_name = None
        contacts = value.get("contacts", [])
        if contacts:
            user_name = contacts[0].get("profile", {}).get("name")

        logger.info(
            "webhook_message_received | phone=%s type=%s message_id=%s name=%s",
            phone,
            msg_type,
            message_id,
            user_name,
        )

        send_typing_indicator(phone, message_id)

        # ── Read session early so we can check step before audio routing ──
        session = state.sessions.get(phone, {"step": "NEW"})

        # ── Voice note — transcribe to text, then route normally ──
        # Skip auto-transcription if the user is in AWAITING_VOICE_STOCK —
        # that step handles the raw audio itself via step_3_handle_voice.
        if msg_type == "audio" and session.get("step") != "AWAITING_VOICE_STOCK":
            media_id = message.get("audio", {}).get("id")
            transcript = (
                await transcribe_voice_message(media_id, phone) if media_id else None
            )
            if not transcript:
                send_text(
                    phone,
                    "I could not understand that voice note. Please try again or type your message.",
                )
                return Response(status_code=200)
            logger.info(
                "voice_transcribed | phone=%s transcript=%r", phone, transcript[:80]
            )
            # Treat the transcript as a regular text message from here on
            msg_type = "text"
            message = {**message, "text": {"body": transcript}}

        # ── Global reset — works at any point in onboarding or after ──
        _raw_text = message.get("text", {}).get("body", "").strip().lower()
        if msg_type == "text" and _raw_text in (
            "delete my data",
            "reset",
            "start over",
        ):
            owner = db.query(Owner).filter(Owner.phone_number == phone).first()
            await onboarding.handle_existing_user(phone, message, owner)
            return Response(status_code=200)

        # ── Existing owner ──
        owner = db.query(Owner).filter(Owner.phone_number == phone).first()
        if owner:
            logger.info("webhook_existing_owner | phone=%s", phone)
            await handle_existing_owner(phone, message, owner)
            return Response(status_code=200)

        # ── New / in-progress user ──
        # session already read above — just ensure fallback is safe
        incoming_text = message.get("text", {}).get("body", "").strip().lower()

        logger.info("webhook_session | phone=%s step=%s", phone, session["step"])

        _mid_onboarding = session["step"] not in ("NEW", "IDLE", "COMPLETE")

        # ── Concurrent webhook skip ──
        if session["step"] in ("PROCESSING_PHOTO", "DAILY_PROCESSING_PHOTO"):
            logger.info(
                "concurrent_webhook_skip | phone=%s step=%s", phone, session["step"]
            )
            return Response(status_code=200)

        # ── Mid-onboarding cancel — restart from welcome at any step ──
        if (
            msg_type == "text"
            and incoming_text in ("cancel", "restart")
            and _mid_onboarding
        ):
            logger.info(
                "webhook_onboarding_cancelled | phone=%s step=%s",
                phone,
                session["step"],
            )
            await onboarding.step_1_greeting_button(phone, name=user_name)
            return Response(status_code=200)

        if session["step"] == "NEW" or (
            incoming_text in ["hi", "hello", "hey", "start"] and not _mid_onboarding
        ):
            await onboarding.step_1_greeting_button(phone, name=user_name)

        elif session["step"] == "AWAITING_BUTTON_CLICK":
            button_id = message.get("interactive", {}).get("button_reply", {}).get("id")
            if button_id == "start_onboarding":
                logger.info("webhook_button_start_onboarding | phone=%s", phone)
                if session.get("name"):
                    await onboarding.step_1b_skip_name_ask_shop(phone)
                else:
                    await step_1b_ask_name(phone)
            else:
                logger.warning(
                    "webhook_unexpected_button | phone=%s button_id=%s",
                    phone,
                    button_id,
                )
                send_reply_buttons(
                    to=phone,
                    body_text="Please tap the button below to get started.",
                    buttons=[{"id": "start_onboarding", "title": "Show me how"}],
                )

        elif session["step"] == "AWAITING_NAME":
            if msg_type != "text" or not incoming_text:
                send_text(phone, "Please type your name to continue.")
            else:
                await step_1c_handle_name(
                    phone, message.get("text", {}).get("body", ""), session
                )

        elif session["step"] == "AWAITING_SHOP":
            if msg_type != "text" or not incoming_text:
                send_text(phone, "Please type your shop name to continue.")
            else:
                await step_1d_handle_shop(
                    phone, message.get("text", {}).get("body", ""), session
                )

        elif session["step"] == "AWAITING_LOCATION":
            if msg_type != "text" or not incoming_text:
                send_text(phone, "Please type your shop location to continue.")
            else:
                await step_1e_handle_location(
                    phone, message.get("text", {}).get("body", ""), session
                )

        elif session["step"] == "AWAITING_CATEGORY":
            row_id = message.get("interactive", {}).get("list_reply", {}).get("id")
            if not row_id:
                send_text(phone, "Please pick a category from the list above.")
            else:
                await step_1f_handle_category(phone, row_id, session)

        elif session["step"] == "AWAITING_INPUT_TYPE":
            # Now uses send_list_message so reads list_reply, with button_reply as fallback
            interactive = message.get("interactive", {})
            button_id = interactive.get("list_reply", {}).get("id") or interactive.get(
                "button_reply", {}
            ).get("id")
            await step_2b_handle_input_choice(phone, button_id)

        elif session["step"] == "AWAITING_TEXT_STOCK":
            if msg_type == "text":
                await step_2c_handle_text_stock(phone, incoming_text, session)
            else:
                send_text(
                    phone,
                    "Please type your stock list as a message.\n\n"
                    "Example: Sneakers: 15, Heels: 10, Bags: 5",
                )

        elif session["step"] == "AWAITING_PHOTO":
            if msg_type == "image":
                await step_3_handle_photo(phone, message, session)
            else:
                # Race condition & Buffering check
                # 1. If we are currently buffering images (list is not empty), ignore non-image messages (e.g. captions)
                if state.sessions.get(phone, {}).get("image_buffer"):
                    logger.info("buffering_skip | phone=%s", phone)
                    return Response(status_code=200)

                # 2. Wait briefly to let concurrent image requests update the state to PROCESSING
                await asyncio.sleep(2.0)

                # 3. Check buffer AND step again
                # If an image arrived while we were sleeping, the buffer will now be populated
                if state.sessions.get(phone, {}).get("image_buffer"):
                    logger.info("buffering_skip_after_sleep | phone=%s", phone)
                    return Response(status_code=200)

                current_step = state.sessions.get(phone, {}).get("step")

                if current_step == "PROCESSING_PHOTO":
                    logger.info(
                        "race_condition_skip | phone=%s step=%s", phone, current_step
                    )
                    return Response(status_code=200)

                logger.warning(
                    "webhook_wrong_message_type | phone=%s step=AWAITING_PHOTO type=%s",
                    phone,
                    msg_type,
                )
                send_text(
                    phone,
                    "Please send a photo of your shelves. 📸\n\n"
                    "Open your camera, take a clear picture of your stock, and send it here.",
                )
        elif session["step"] == "AWAITING_VOICE_STOCK":
            if msg_type == "audio":
                await step_3_handle_voice(phone, message, session)
            else:
                send_text(
                    phone,
                    "Please send a voice note. 🎤\n\nOr type *cancel* to go back.",
                )

        elif session["step"] == "AWAITING_FLOW":
            interactive = message.get("interactive", {})
            flow_response = interactive.get("nfm_reply", {})
            button_id = interactive.get("button_reply", {}).get("id")

            if flow_response:
                await step_5_handle_flow_submission(phone, flow_response, session)
            elif button_id == "inventory_correct":
                logger.info("webhook_inventory_confirmed_button | phone=%s", phone)
                await onboarding.step_5b_ask_stock_value(phone)
            elif button_id == "inventory_edit":
                logger.info("webhook_inventory_edit_button | phone=%s", phone)
                send_text(
                    phone,
                    "No problem. What needs changing?\n\n"
                    "You can say things like:\n"
                    '• "Sneakers are actually 20"\n'
                    '• "Rice is 500 cedis"\n'
                    '• "Remove the milk"\n\n'
                    "Type your correction below. 👇",
                )
                state.sessions[phone]["step"] = "AWAITING_CORRECTION"
            else:
                logger.warning(
                    "webhook_unexpected_flow_input | phone=%s type=%s", phone, msg_type
                )
                send_reply_buttons(
                    to=phone,
                    body_text="Please tap a button below to continue.",
                    buttons=[
                        {"id": "inventory_correct", "title": "Yes, correct ✅"},
                        {"id": "inventory_edit", "title": "No, edit it"},
                    ],
                )

        elif session["step"] == "AWAITING_CORRECTION":
            if msg_type == "text":
                await onboarding.step_4b_handle_correction(
                    phone, incoming_text, session
                )
            else:
                send_text(phone, "Please type your correction as text.")

        elif session["step"] == "AWAITING_CAPS_Q1":
            if msg_type != "text":
                send_text(phone, "Please reply with a number. Example: *50000*")
            else:
                await step_5b_handle_stock_value(phone, incoming_text, session)

        elif session["step"] == "AWAITING_CAPS_Q2":
            if msg_type != "text":
                send_text(phone, "Please reply with a number. Example: *10000*")
            else:
                await step_5c_handle_restart_cap(phone, incoming_text, session)

        elif session["step"] in ("AWAITING_PAYMENT_DECISION", "IDLE", "COMPLETE"):
            await onboarding.handle_existing_user(phone, message, None)

        else:
            logger.warning(
                "webhook_unhandled_step | phone=%s step=%s", phone, session["step"]
            )

        return Response(status_code=200)

    except Exception:
        logger.exception("webhook_error | phone=%s", phone)
        if phone:
            try:
                send_text(
                    phone,
                    "Something went wrong on our end. Please try again in a moment.\n\n"
                    "If this keeps happening, type *cancel* to restart.",
                )
            except Exception:
                logger.exception("webhook_error_notify_failed | phone=%s", phone)
        return Response(status_code=200)  # always 200 so WhatsApp doesn't retry


async def handle_existing_owner(phone, message, owner):
    await onboarding.handle_existing_user(phone, message, owner)


def _log_message_status(status: dict) -> None:
    """
    Log all important fields from a WhatsApp status update.

    Possible statuses:
        sent      — message left Meta's servers
        delivered — message reached the user's device
        read      — user opened the message
        failed    — message could not be delivered
    """
    message_id = status.get("id")
    status_state = status.get("status")
    recipient = status.get("recipient_id")
    timestamp = status.get("timestamp")
    pricing = status.get("pricing", {})
    errors = status.get("errors", [])
    conversation = status.get("conversation", {})

    if status_state == "failed":
        error_info = errors[0] if errors else {}
        logger.error(
            "message_status_failed | message_id=%s recipient=%s "
            "timestamp=%s error_code=%s error_title=%s error_message=%s",
            message_id,
            recipient,
            timestamp,
            error_info.get("code"),
            error_info.get("title"),
            error_info.get("message"),
        )
    else:
        logger.info(
            "message_status | id=%s status=%s recipient=%s timestamp=%s "
            "billable=%s category=%s conversation_id=%s",
            message_id,
            status_state,
            recipient,
            timestamp,
            pricing.get("billable"),
            pricing.get("category"),
            conversation.get("id"),
        )


def _flow_response(screen: str, data: dict) -> dict:
    return {"screen": screen, "data": data}


def _flow_error(message: str) -> dict:
    return {"screen": "ERROR", "data": {"error_message": message}}
