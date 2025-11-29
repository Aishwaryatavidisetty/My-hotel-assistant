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
# Using standard text avatars or None to let CSS handle it
USER_AVATAR = None 
BOT_AVATAR = None

def _init_app_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "booking_state" not in st.session_state:
        st.session_state.booking_state = BookingState()
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = None
    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks = []

# --- PROFESSIONAL CSS STYLING ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* --- 1. Global Reset & Fonts --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1f2937;
        }
        
        .stApp {
            background-color: #f3f4f6; /* Light Gray Background */
        }

        /* --- 2. Chat Container --- */
        /* Make chat container white and centered */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 800px;
        }

        /* --- 3. Chat Messages --- */
        
        /* User Message (Right, Blue) */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            flex-direction: row-reverse;
            background-color: transparent;
            border: none;
            margin-bottom: 8px;
        }
        
        div[data-testid="stChatMessage"]:nth-child(odd) .stChatMessageContent {
            background-color: #2563eb; /* Royal Blue */
            color: white !important;
            border-radius: 20px 20px 4px 20px;
            padding: 12px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            max-width: 80%;
            text-align: left;
        }
        
        /* Force white text in user bubbles */
        div[data-testid="stChatMessage"]:nth-child(odd) p {
            color: white !important;
            margin: 0;
        }

        /* Assistant Message (Left, White) */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background-color: transparent;
            border: none;
            margin-bottom: 8px;
        }
        
        div[data-testid="stChatMessage"]:nth-child(even) .stChatMessageContent {
            background-color: #ffffff;
            color: #1f2937;
            border-radius: 20px 20px 20px 4px;
            padding: 12px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            border: 1px solid #e5e7eb;
            max-width: 80%;
        }

        /* Hide Default Avatars to look cleaner */
        [data-testid="stChatMessageAvatarBackground"] {
            display: none;
        }

        /* --- 4. Fixed Input Area --- */
        /* We style the audio input to float nicely at the bottom right */
        
        .stAudioInput {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            width: 50px;
            height: 50px;
        }
        
        /* Style the Mic Button to be a floating circle */
        .stAudioInput button {
            background-color: #2563eb;
            color: white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            border: none;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
            transition: transform 0.1s;
        }
        
        .stAudioInput button:hover {
            transform: scale(1.05);
            background-color: #1d4ed8;
        }
        
        /* Hide audio label */
        .stAudioInput label {
            display: none;
        }

        /* --- 5. Clean Up --- */
        /* Hide standard header */
        header {visibility: hidden;}
        
        /* Hide footer */
        footer {visibility: hidden;}
        
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
        page_title="Hotel AI",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    cfg = load_config()
    _init_app_state()

    # Sidebar just for Admin access
    with st.sidebar:
        st.title("Settings")
        menu = st.radio("Mode", ["Chat", "Admin Dashboard"])

    if menu == "Chat":
        run_chat_assistant(cfg)
    else:
        render_admin_dashboard()


def run_chat_assistant(cfg):
    # Header
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.write("") # Spacer
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color: #111827; margin-bottom: 0;'>Grand Hotel Concierge</h1>
            <p style='color: #6b7280; font-size: 0.9rem;'>Your personal AI assistant for bookings & services</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat History
    # We use a container to hold messages
    chat_placeholder = st.container()

    with chat_placeholder:
        if not st.session_state.messages:
            st.markdown("""
            <div style='background-color: #e0f2fe; padding: 15px; border-radius: 10px; text-align: center; color: #0369a1; margin: 20px auto; max-width: 600px;'>
                👋 <b>Welcome!</b><br>
                Try asking: <i>"I want to book a room"</i> or <i>"What are the check-in times?"</i>
            </div>
            """, unsafe_allow_html=True)
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --- INPUT AREA ---
    # 1. Text Input (Standard)
    user_input = st.chat_input("Type a message...")
    
    # 2. Audio Input (Floating Button via CSS)
    audio_val = st.audio_input("Voice", label_visibility="collapsed")

    # --- LOGIC ---
    final_input = None
    input_source = "text"

    if audio_val:
        with st.spinner("🎧 Processing..."):
            transcribed = transcribe_audio(audio_val)
            if transcribed:
                final_input = transcribed
                input_source = "audio"
    
    if user_input:
        final_input = user_input
        input_source = "text"

    if not final_input:
        return

    # Show User Message
    store_message(st.session_state.messages, "user", final_input)
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(final_input)

    # --- INTENT & RESPONSE ---
    detected_intent = detect_intent(final_input)
    final_intent = detected_intent
    
    rag_keywords = ["price", "cost", "rate", "wifi", "pool", "gym", "check-in", "policy", "refund", "breakfast", "location"]
    check_booking_keywords = ["check booking", "status", "my booking"]
    
    if any(kw in final_input.lower() for kw in check_booking_keywords):
        final_intent = "check_booking"
    elif any(kw in final_input.lower() for kw in rag_keywords):
        final_intent = "faq_rag"
    elif st.session_state.booking_state.active:
        if "cancel" in final_input.lower():
            final_intent = "booking"
        elif detected_intent == "faq_rag": 
             final_intent = "faq_rag"
        else:
            final_intent = "booking"

    # Generate Answer
    if final_intent == "booking":
        response_text = handle_booking_intent(cfg, final_input)
    elif final_intent == "check_booking":
        response_text = handle_check_booking(final_input)
    elif final_intent == "faq_rag":
        response_text = handle_faq_intent(final_input)
    elif final_intent == "small_talk":
        response_text = "Hello! I can help you with room bookings or hotel information."
    else:
        response_text = "I'm not sure I understand. Would you like to make a booking?"

    # Display Assistant Response
    store_message(st.session_state.messages, "assistant", response_text)
    with chat_placeholder:
        with st.chat_message("assistant"):
            st.markdown(response_text)
            
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
            return f"No bookings found for **{email}**."
        
        msg = f"**Found {len(results)} booking(s):**\n"
        for b in results:
            msg += f"\n🔸 **{b['type']} Room** on {b['date']} ({b['status']})"
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
        return "🚫 Booking cancelled."

    if state.awaiting_confirmation:
        if "confirm" in lower_msg or lower_msg in ("yes", "yes, confirm"):
            payload = state.to_payload()
            result = booking_persistence_tool(cfg, payload)

            if not result["success"]:
                st.session_state.booking_state = BookingState()
                return f"⚠️ Error: {result['error']}"

            booking_id = result["booking_id"]
            email_body = (
                f"Booking ID: {booking_id}\n\n{generate_confirmation_text(state)}"
            )

            email_result = email_tool(
                cfg,
                to_email=state.email,
                subject="Hotel Booking Confirmation",
                body=email_body,
            )
            
            msg = f"🎉 **Success!** Your booking ID is `{booking_id}`."
            st.session_state.booking_state = BookingState()
            return msg

        return "Please type **'confirm'** to finish."

    state = update_state_from_message(user_message, state)
    st.session_state.booking_state = state

    if state.errors:
        field, msg = next(iter(state.errors.items()))
        return f"⚠️ {msg}"

    missing = get_missing_fields(state)

    if missing:
        return next_question_for_missing_field(missing[0])

    summary = generate_confirmation_text(state)
    state.awaiting_confirmation = True
    st.session_state.booking_state = state

    return f"**Please confirm details:**\n\n{summary}\n\nType **'confirm'**."


def handle_faq_intent(user_message: str) -> str:
    store: RAGStore = st.session_state.rag_store
    if store is None or store.size == 0:
        return "I can't answer that yet. Please upload hotel documents first."
    else:
        return rag_tool(store, user_message)


if __name__ == "__main__":
    main()