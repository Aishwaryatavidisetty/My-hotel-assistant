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
        /* --- Global App Styling --- */
        .stApp {
            background-color: #ffffff;
        }
        
        /* --- Chat Messages --- */
        /* User Message (Right, Blue) */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            flex-direction: row-reverse;
            background-color: transparent;
            border: none;
            margin-bottom: 8px;
        }
        
        div[data-testid="stChatMessage"]:nth-child(odd) .stChatMessageContent {
            background-color: #007bff;
            color: white !important;
            border-radius: 18px 18px 0px 18px;
            padding: 10px 15px;
            margin-bottom: 5px;
            border: none;
            width: fit-content;
            max-width: 80%;
            margin-left: auto; /* Push to right */
        }
        /* Force text color in user bubble */
        div[data-testid="stChatMessage"]:nth-child(odd) p {
            color: white !important;
        }
        div[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stChatMessageAvatarBackground"] {
            display: none; /* Hide avatar for cleaner look, or keep if preferred */
        }

        /* Assistant Message (Gray, Left aligned) */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #f1f0f0;
            color: black;
            border-radius: 18px 18px 18px 0px;
            padding: 10px 15px;
            margin-bottom: 5px;
            border: none;
            width: fit-content;
            max-width: 80%;
        }
        
        /* --- Integrated Audio Input Styling --- */
        /* This hacks the audio widget to float near the chat input */
        .stAudioInput {
            position: fixed;
            bottom: 10px; /* Adjust based on chat input height */
            left: 20px;   /* Position to the left of the text box */
            z-index: 1000;
            width: 40px;  /* Small width to look like a button */
            height: 40px;
        }
        
        /* Hide the label and extra spacing of audio widget */
        .stAudioInput > label {
            display: none;
        }
        
        /* Make the internal button round and icon-like */
        .stAudioInput button {
            border-radius: 50%;
            width: 40px;
            height: 40px;
            background-color: #f0f2f6;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Adjust Chat Input to make room for the mic on the left (if possible) 
           Note: Streamlit doesn't easily allow padding left on chat_input via CSS class directly 
           without affecting other elements, so we float the mic on top-left or right.
        */
        
        /* Clean up header */
        header {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: AUDIO TRANSCRIPTION (STT) ---
def transcribe_audio(audio_file):
    try:
        if "google" in st.secrets:
            api_key = st.secrets["google"]["api_key"]
        elif "gemini" in st.secrets:
            api_key = st.secrets["gemini"]["api_key"]
        else:
            api_key = st.secrets.get("google_api_key", "")
        genai.configure(api_key=api_key)
        
        audio_bytes = audio_file.read()

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
        initial_sidebar_state="collapsed" # Hide sidebar for cleaner look
    )
    
    inject_custom_css()
    cfg = load_config()
    _init_app_state()

    # Minimal Sidebar
    with st.sidebar:
        st.title("Menu")
        menu = st.radio("Go to", ["Chat Assistant", "Admin Dashboard"])
        st.divider()
        st.info("💡 You can speak to the assistant using the floating microphone!")

    if menu == "Chat Assistant":
        run_chat_assistant(cfg)
    else:
        render_admin_dashboard()


def run_chat_assistant(cfg):
    # Minimal Header
    col1, col2 = st.columns([0.5, 8])
    with col1:
        st.write("🏨")
    with col2:
        st.markdown("### Grand Hotel Concierge")

    # Hidden File Uploader
    with st.expander("⚙️ Admin Settings (Upload PDFs)"):
        uploaded_files = st.file_uploader(
            "Upload policies",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Update Knowledge Base"):
            with st.spinner("Processing..."):
                rag_store, chunks = build_rag_store_from_uploads(
                    uploaded_files, RAGConfig()
                )
                st.session_state.rag_store = rag_store
                st.session_state.rag_chunks = chunks
            st.success(f"Indexed {len(chunks)} chunks.")

    # --- CHAT AREA ---
    # Max height container to keep chat scrolling nicely above inputs
    chat_container = st.container(height=600)

    with chat_container:
        if not st.session_state.messages:
            st.markdown("*How can I help you today?*")
            
        for msg in st.session_state.messages:
            # We removed avatars in CSS for a cleaner "text message" look on User side
            # But we keep the structure
            with st.chat_message(msg["role"], avatar=BOT_AVATAR if msg["role"]=="assistant" else USER_AVATAR):
                st.write(msg["content"])

    # --- INPUTS ---
    user_input = None
    input_source = "text"
    
    # 1. Voice Input (Floating via CSS)
    audio_val = st.audio_input("Voice", label_visibility="collapsed")
    
    # 2. Text Input (Standard Streamlit Bottom Bar)
    text_val = st.chat_input("Message...")

    # Logic
    if audio_val:
        with st.spinner("🎧 Processing voice..."):
            transcribed = transcribe_audio(audio_val)
            if transcribed:
                user_input = transcribed
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

    # --- INTENT & RESPONSE ---
    detected_intent = detect_intent(user_input)
    final_intent = detected_intent
    
    rag_keywords = ["price", "cost", "rate", "wifi", "pool", "gym", "check-in", "policy", "refund", "breakfast", "location"]
    check_booking_keywords = ["check booking", "status", "my booking"]
    # NEW: Keywords to catch affirmation to start a booking
    start_booking_keywords = ["book", "yes", "sure", "yep", "confirm", "reservation"]
    
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
    # NEW: Check if user said "yes" or "book" to start a booking (if not active)
    elif any(kw in user_input.lower() for kw in start_booking_keywords):
        final_intent = "booking"

    # Generate Answer
    if final_intent == "booking":
        response_text = handle_booking_intent(cfg, user_input)
    elif final_intent == "check_booking":
        response_text = handle_check_booking(user_input)
    elif final_intent == "faq_rag":
        response_text = handle_faq_intent(user_input)
    elif final_intent == "small_talk":
        response_text = "Hello! I can help you with room bookings or hotel information."
    else:
        response_text = "I'm not sure I understood. Would you like to make a booking?"

    # Display Assistant Response
    store_message(st.session_state.messages, "assistant", response_text)
    with chat_container:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.write(response_text)
            
            # Smart Voice Response
            if input_source == "audio":
                text_to_speech(response_text)


def handle_check_booking(user_input: str) -> str:
    import re
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', user_input)
    if email_match:
        email = email_match.group(0)
        results = find_booking_by_email(email)
        if not results:
            return f"No bookings found for {email}."
        
        msg = f"**Bookings for {email}:**\n"
        for b in results:
            msg += f"\n- **{b['type']} Room** on {b['date']} ({b['status']})"
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
        return "Booking cancelled."

    if state.awaiting_confirmation:
        if "confirm" in lower_msg or lower_msg in ("yes", "yes, confirm"):
            payload = state.to_payload()
            result = booking_persistence_tool(cfg, payload)

            if not result["success"]:
                st.session_state.booking_state = BookingState()
                return f"Error: {result['error']}"

            booking_id = result["booking_id"]
            email_body = (
                f"Your booking is confirmed!\n\n"
                f"Booking ID: {booking_id}\n\n"
                f"{generate_confirmation_text(state)}"
            )

            email_result = email_tool(
                cfg,
                to_email=state.email,
                subject="Hotel Booking Confirmation",
                body=email_body,
            )
            
            msg = f"🎉 **Confirmed!** ID: `{booking_id}`"
            st.session_state.booking_state = BookingState()
            return msg

        return "Type **'confirm'** to finish."

    state = update_state_from_message(user_message, state)
    st.session_state.booking_state = state

    if state.errors:
        field, msg = next(iter(state.errors.items()))
        return msg

    missing = get_missing_fields(state)

    if missing:
        return next_question_for_missing_field(missing[0])

    summary = generate_confirmation_text(state)
    state.awaiting_confirmation = True
    st.session_state.booking_state = state

    return f"**Confirm details:**\n\n{summary}\n\nType **'confirm'**."


def handle_faq_intent(user_message: str) -> str:
    store: RAGStore = st.session_state.rag_store
    if store is None or store.size == 0:
        return "I can't answer that yet. Please upload documents first."
    else:
        return rag_tool(store, user_message)


if __name__ == "__main__":
    main()