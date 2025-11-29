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

# --- CSS STYLING ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* --- Global Settings --- */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* --- Chat Container Styling --- */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }

        /* --- User Message Bubble (Right-ish / Distinct Color) --- */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background: linear-gradient(135deg, #0062cc 0%, #004a99 100%);
            color: white;
            border-radius: 16px 16px 0px 16px;
            padding: 1rem;
            margin-bottom: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border: none;
        }
        
        /* Fix text color inside user bubble */
        div[data-testid="stChatMessage"]:nth-child(odd) p {
            color: white !important;
        }
        
        /* Fix Avatar Background for User */
        div[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stChatMessageAvatarBackground"] {
            background-color: #004a99;
        }

        /* --- Assistant Message Bubble (Left / Clean White) --- */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #ffffff;
            border-radius: 16px 16px 16px 0px;
            padding: 1rem;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid #eef0f3;
        }

        /* --- Audio Input Styling --- */
        .stAudioInput {
            position: fixed;
            bottom: 80px; /* Sits right above the chat input */
            z-index: 1000;
            width: 100%;
            background: transparent;
        }
        
        /* Make the audio widget compact */
        .stAudioInput > div {
            background-color: transparent !important;
            border: none !important;
        }

        /* --- Header Styling --- */
        h1 {
            color: #1e3a8a;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }
        
        /* --- Sidebar Styling --- */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
    </style>
    """, unsafe_allow_html=True)

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

        # Fallback Strategy
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
                response = model.generate_content([
                    "Transcribe this audio exactly as it is spoken.",
                    {"mime_type": "audio/wav", "data": audio_bytes}
                ])
                return response.text.strip()
            except Exception as e:
                last_error = e
                continue
        
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
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    cfg = load_config()
    _init_app_state()

    # Sidebar Navigation
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("", ["Chat Assistant", "Admin Dashboard"], label_visibility="collapsed")
        st.divider()
        st.info("ℹ️ **Tip:** You can speak to the assistant using the microphone icon!")

    if menu == "Chat Assistant":
        run_chat_assistant(cfg)
    else:
        render_admin_dashboard()


def run_chat_assistant(cfg):
    # Header Section
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=60) # Generic Hotel Icon
    with col2:
        st.title("Grand Hotel AI Concierge")
        st.caption("Book rooms, check status, and get answers instantly.")

    # PDF Upload Expander (Hidden by default to clean up UI)
    with st.expander("📂 Upload Hotel Documents (Admin Only)"):
        uploaded_files = st.file_uploader(
            "Upload policies, room details, etc.",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Update Knowledge Base"):
            with st.spinner("Processing documents..."):
                rag_store, chunks = build_rag_store_from_uploads(
                    uploaded_files, RAGConfig()
                )
                st.session_state.rag_store = rag_store
                st.session_state.rag_chunks = chunks
            st.success(f"Indexed {len(chunks)} chunks successfully.")

    st.divider()

    # --- CHAT CONTAINER ---
    # Calculates height to leave space for fixed input at bottom
    chat_container = st.container(height=550)

    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Hi there! I can help you book a room or answer questions about our hotel.")
            
        for msg in st.session_state.messages:
            avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

    # --- INPUT AREA ---
    user_input = None
    input_source = "text"
    
    # 1. Audio Input (Sits visually above text input)
    # The 'label_visibility="collapsed"' makes it cleaner
    audio_val = st.audio_input("Speak", label_visibility="collapsed")
    
    # 2. Text Input
    text_val = st.chat_input("Type your message here...")

    # Logic to prioritize inputs
    if audio_val:
        with st.spinner("🎧 Listening..."):
            transcribed_text = transcribe_audio(audio_val)
            if transcribed_text:
                user_input = transcribed_text
                input_source = "audio"

    if text_val:
        user_input = text_val
        input_source = "text"

    if not user_input:
        return

    # --- UI UPDATE ---
    with chat_container:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(user_input)
    
    store_message(st.session_state.messages, "user", user_input)

    # --- INTENT & ROUTING ---
    detected_intent = detect_intent(user_input)
    final_intent = detected_intent
    
    # Simple Keyword Overrides
    rag_keywords = ["price", "cost", "rate", "wifi", "pool", "gym", "check-in", "policy", "refund", "breakfast", "location"]
    check_booking_keywords = ["check booking", "status", "my booking"]
    
    if any(kw in user_input.lower() for kw in check_booking_keywords):
        final_intent = "check_booking"
    elif any(kw in user_input.lower() for kw in rag_keywords):
        final_intent = "faq_rag"
    elif st.session_state.booking_state.active:
        if "cancel" in user_input.lower():
            final_intent = "booking"
        elif detected_intent == "faq_rag": 
             final_intent = "faq_rag"
        else:
            final_intent = "booking"

    # --- GENERATE RESPONSE ---
    with st.spinner("thinking..."):
        if final_intent == "booking":
            response_text = handle_booking_intent(cfg, user_input)
        elif final_intent == "check_booking":
            response_text = handle_check_booking(user_input)
        elif final_intent == "faq_rag":
            response_text = handle_faq_intent(user_input)
        elif final_intent == "small_talk":
            response_text = "Hello! I'm your hotel concierge. I can help with bookings, existing reservations, or hotel information."
        else:
            response_text = "I didn't quite catch that. Could you clarify if you want to book a room or ask a question?"

        # --- RESPOND & SPEAK IF NEEDED ---
        store_message(st.session_state.messages, "assistant", response_text)
        with chat_container:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.write(response_text)
                
                # Smart Voice Logic:
                # Only speak if the input came from Audio (Auto-Mode)
                if input_source == "audio":
                    text_to_speech(response_text)


def handle_check_booking(user_input: str) -> str:
    import re
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', user_input)
    if email_match:
        email = email_match.group(0)
        results = find_booking_by_email(email)
        if not results:
            return f"I searched for {email} but couldn't find any active bookings."
        
        msg = f"**Found {len(results)} booking(s) for {email}:**\n"
        for b in results:
            msg += f"\n🆔 **ID:** `{b['booking_id']}`\n🛏️ **Type:** {b['type']}\n📅 **Date:** {b['date']} at {b['time']}\n✅ **Status:** {b['status']}\n---"
        return msg
    else:
        return "Please provide your email address so I can look up your booking."


def handle_booking_intent(cfg, user_message: str) -> str:
    state: BookingState = st.session_state.booking_state
    
    if not state.active:
        state.active = True
        st.session_state.booking_state = state

    lower_msg = user_message.strip().lower()

    if "cancel" in lower_msg:
        st.session_state.booking_state = BookingState()
        return "Booking cancelled. Let me know when you're ready to try again."

    if state.awaiting_confirmation:
        if "confirm" in lower_msg or lower_msg in ("yes", "yes, confirm"):
            payload = state.to_payload()
            result = booking_persistence_tool(cfg, payload)

            if not result["success"]:
                st.session_state.booking_state = BookingState()
                return f"⚠️ Error saving booking: {result['error']}"

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
            
            msg = f"🎉 **Success!** Your booking is confirmed.\n\n🆔 **Booking ID:** `{booking_id}`"
            if not email_result["success"]:
                msg += f"\n\n(Note: Confirmation email failed to send, but your booking is saved.)"
            else:
                msg += "\n\n📧 A confirmation email has been sent to your inbox."

            st.session_state.booking_state = BookingState()
            return msg

        return "Please type **'confirm'** to finalize your booking, or **'cancel'** to stop."

    state = update_state_from_message(user_message, state)
    st.session_state.booking_state = state

    if state.errors:
        field, msg = next(iter(state.errors.items()))
        return f"⚠️ {msg}"

    missing = get_missing_fields(state)

    if missing:
        next_field = missing[0]
        return next_question_for_missing_field(next_field)

    summary = generate_confirmation_text(state)
    state.awaiting_confirmation = True
    st.session_state.booking_state = state

    return (
        "**Please confirm your details:**\n\n"
        f"{summary}\n\n"
        "Type **'confirm'** to finish."
    )


def handle_faq_intent(user_message: str) -> str:
    store: RAGStore = st.session_state.rag_store
    if store is None or store.size == 0:
        return "⚠️ I can't answer that yet. Please upload the hotel policy documents in the sidebar first."
    else:
        return rag_tool(store, user_message)


# Legacy respond function not used in new flow
def respond(text: str, enable_voice: bool = False):
    store_message(st.session_state.messages, "assistant", text)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.write(text)
        if enable_voice:
            text_to_speech(text)


if __name__ == "__main__":
    main()