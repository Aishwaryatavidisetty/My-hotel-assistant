from __future__ import annotations

import sys
import os
import io
from gtts import gTTS  # Text to Speech

# --- Add project root to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st
import google.generativeai as genai

# IMPORTS
from config import load_config
from chat_logic import detect_intent, store_message
from rag_pipeline import RAGStore, RAGConfig, build_rag_store_from_uploads
from rag_pipeline import rag_tool 
from tools import booking_persistence_tool, email_tool, find_booking_by_email
from admin_dashboard import render_admin_dashboard

from booking_flow import (
    BookingState,
    get_missing_fields,
    generate_confirmation_text,
    update_state_from_message,
    next_question_for_missing_field,
)

# --- CONSTANTS FOR UX ---
USER_AVATAR = "👤"
BOT_AVATAR = "🏨"

def _init_app_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "booking_state" not in st.session_state:
        st.session_state.booking_state = BookingState()
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = None
    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks = []

# --- HELPER: AUDIO TRANSCRIPTION (STT) ---
def transcribe_audio(audio_file):
    try:
        # Configure genai (ensure key is loaded)
        if "google" in st.secrets:
            api_key = st.secrets["google"]["api_key"]
        elif "gemini" in st.secrets:
            api_key = st.secrets["gemini"]["api_key"]
        else:
            api_key = st.secrets.get("google_api_key", "")
        genai.configure(api_key=api_key)
        
        # Read file bytes
        audio_bytes = audio_file.read()

        # --- UPDATED FALLBACK STRATEGY FOR AUDIO ---
        models_to_try = [
            'gemini-2.0-flash', 
            'gemini-2.0-flash-lite',
            'gemini-1.5-flash',
            'gemini-flash-latest'
        ]
        
        last_error = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                # Gemini expects parts. We can send data directly.
                response = model.generate_content([
                    "Transcribe this audio exactly as it is spoken.",
                    {"mime_type": "audio/wav", "data": audio_bytes}
                ])
                return response.text.strip()
            except Exception as e:
                last_error = e
                continue
        
        # If loop finishes without returning
        st.error(f"Audio transcription failed on all models. Last error: {last_error}")
        return None

    except Exception as e:
        st.error(f"Audio transcription critical error: {e}")
        return None

# --- HELPER: TEXT TO SPEECH (TTS) ---
def text_to_speech(text):
    try:
        if not text: return
        tts = gTTS(text=text, lang='en')
        # Save to memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        st.audio(audio_buffer, format="audio/mp3", start_time=0)
    except Exception as e:
        print(f"TTS Error: {e}")


def main():
    st.set_page_config(
        page_title="AI Hotel Booking Assistant",
        page_icon="🏨",
        layout="wide",
    )

    cfg = load_config()
    _init_app_state()

    menu = st.sidebar.radio("Navigation", ["Chat Assistant", "Admin Dashboard"])
    
    if menu == "Chat Assistant":
        run_chat_assistant(cfg)
    else:
        render_admin_dashboard()


def run_chat_assistant(cfg):
    st.title("🏨 AI Hotel Booking Assistant")

    st.subheader("Upload Hotel PDFs for RAG")
    uploaded_files = st.file_uploader(
        "Upload one or more hotel-related PDFs (policies, room details, etc.)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Build Knowledge Base from PDFs"):
            with st.spinner("Processing and indexing PDFs..."):
                rag_store, chunks = build_rag_store_from_uploads(
                    uploaded_files, RAGConfig()
                )
                st.session_state.rag_store = rag_store
                st.session_state.rag_chunks = chunks
            st.success(
                f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)."
            )

    # --- CHAT INTERFACE LAYOUT ---
    # We use a fixed-height container for messages so the inputs stay at the bottom.
    chat_container = st.container(height=500)

    # Display chat history INSIDE the container
    with chat_container:
        for msg in st.session_state.messages:
            avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

    # --- INPUT AREA (Pinned near bottom) ---
    user_input = None
    
    # 1. Voice Response Toggle (Right above the mic)
    # This places a simple switch near the input area
    enable_voice_response = st.toggle("📣 Read responses aloud", value=False)
    
    # 2. Audio Input
    audio_val = st.audio_input("🎤 Speak to the assistant")
    
    # 3. Text Input
    text_val = st.chat_input("How can I help you with your hotel stay today?")

    if audio_val:
        with st.spinner("Transcribing voice..."):
            transcribed_text = transcribe_audio(audio_val)
            if transcribed_text:
                user_input = transcribed_text

    if text_val:
        user_input = text_val

    if not user_input:
        return

    # --- UI FIX: Display User Message Immediately ---
    # We write this into the container so it appears in history instantly
    with chat_container:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(user_input)
    
    store_message(st.session_state.messages, "user", user_input)

    # --- INTELLIGENT ROUTING LOGIC ---
    
    detected_intent = detect_intent(user_input)
    final_intent = detected_intent
    
    rag_keywords = [
        "price", "cost", "how much", "rate", 
        "wifi", "internet", "pool", "gym", "spa", "parking",
        "check-in", "check-out", "policy", "refund", "cancel",
        "breakfast", "food", "restaurant", "location", "near"
    ]
    
    # Check for Booking Status keywords
    check_booking_keywords = ["check booking", "status", "my booking", "booking details"]
    if any(kw in user_input.lower() for kw in check_booking_keywords):
        final_intent = "check_booking"

    # Priority Routing
    if any(kw in user_input.lower() for kw in rag_keywords):
        final_intent = "faq_rag"
    elif st.session_state.booking_state.active:
        if "cancel" in user_input.lower():
            final_intent = "booking"
        elif detected_intent == "faq_rag": 
             final_intent = "faq_rag"
        else:
            final_intent = "booking"

    # Dispatch with Thinking State
    with st.spinner("Thinking..."):
        if final_intent == "booking":
            response_text = handle_booking_intent(cfg, user_input)
        elif final_intent == "check_booking":
            response_text = handle_check_booking(user_input)
        elif final_intent == "faq_rag":
            response_text = handle_faq_intent(user_input)
        elif final_intent == "small_talk":
            response_text = "Hello! I can help you book rooms, check your booking status, or answer questions about the hotel."
        else:
            response_text = "I’m not sure I understood. Are you trying to make a booking, check a booking, or ask about hotel details?"

        # Display Response
        # We write directly to the container and pass the toggle value
        store_message(st.session_state.messages, "assistant", response_text)
        with chat_container:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.write(response_text)
                if enable_voice_response:
                    text_to_speech(response_text)


def handle_check_booking(user_input: str) -> str:
    # Simple extraction of email-like patterns
    import re
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', user_input)
    
    if email_match:
        email = email_match.group(0)
        results = find_booking_by_email(email)
        if not results:
            return f"I couldn't find any active bookings for {email}."
        
        msg = f"Found {len(results)} booking(s) for {email}:\n"
        for b in results:
            msg += f"\n- **ID:** {b['booking_id']}\n  **Type:** {b['type']}\n  **Date:** {b['date']} at {b['time']}\n  **Status:** {b['status']}\n"
        return msg
    else:
        return "Please provide your email address to check your booking."


def handle_booking_intent(cfg, user_message: str) -> str:
    state: BookingState = st.session_state.booking_state
    
    if not state.active:
        state.active = True
        st.session_state.booking_state = state

    lower_msg = user_message.strip().lower()

    if "cancel" in lower_msg:
        st.session_state.booking_state = BookingState()
        return "Booking cancelled. Let me know if you'd like to start again."

    if state.awaiting_confirmation:
        if "confirm" in lower_msg or lower_msg in ("yes", "yes, confirm"):
            payload = state.to_payload()
            result = booking_persistence_tool(cfg, payload)

            if not result["success"]:
                st.session_state.booking_state = BookingState()
                return f"Error saving booking: {result['error']}"

            booking_id = result["booking_id"]
            email_body = (
                "Your hotel booking is confirmed.\n\n"
                f"Booking ID: {booking_id}\n\n"
                f"{generate_confirmation_text(state)}"
            )

            email_result = email_tool(
                cfg,
                to_email=state.email,
                subject="Hotel Booking Confirmation",
                body=email_body,
            )
            
            msg = f"🎉 Booking confirmed! ID: {booking_id}. "
            if not email_result["success"]:
                msg += f" (Email failed: {email_result['error']})"
            else:
                msg += " A confirmation email has been sent."

            st.session_state.booking_state = BookingState()
            return msg

        return "Type 'confirm' to finalize or 'cancel' to stop."

    state = update_state_from_message(user_message, state)
    st.session_state.booking_state = state

    if state.errors:
        field, msg = next(iter(state.errors.items()))
        return msg

    missing = get_missing_fields(state)

    if missing:
        next_field = missing[0]
        return next_question_for_missing_field(next_field)

    summary = generate_confirmation_text(state)
    state.awaiting_confirmation = True
    st.session_state.booking_state = state

    return (
        "Here are your booking details:\n\n"
        f"{summary}\n"
        "Type **'confirm'** to finalize or **'cancel'**."
    )


def handle_faq_intent(user_message: str) -> str:
    store: RAGStore = st.session_state.rag_store

    if store is None or store.size == 0:
        return "No hotel documents indexed yet. Upload PDFs and click 'Build Knowledge Base', then ask your question again."
    else:
        return rag_tool(store, user_message)


# Modified respond function is now handled inline within run_chat_assistant
# to ensure it prints inside the chat_container.
# We keep this for non-chat usage if any.
def respond(text: str, enable_voice: bool = False):
    store_message(st.session_state.messages, "assistant", text)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.write(text)
        if enable_voice:
            text_to_speech(text)


if __name__ == "__main__":
    main()