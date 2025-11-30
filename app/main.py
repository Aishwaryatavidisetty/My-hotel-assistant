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
        /* --- Audio Input Styling (Floating Bottom Right) --- */
        .stAudioInput {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            width: 50px;
            height: 50px;
        }
        
        /* Style the Mic Button */
        .stAudioInput button {
            background-color: #2563eb;
            color: white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            border: none;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
        }
        
        /* Hover Effect */
        .stAudioInput button:hover {
            background-color: #1d4ed8;
            transform: scale(1.05);
        }
        
        /* Hide label */
        .stAudioInput label {
            display: none;
        }

        /* --- Hide Header/Footer for clean look --- */
        header {visibility: hidden;}
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
        page_title="AI Hotel Booking Assistant",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="expanded" # Restored Sidebar
    )
    
    inject_custom_css()
    cfg = load_config()
    _init_app_state()

    # --- SIDEBAR NAVIGATION (Restored) ---
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Go to", ["Chat Assistant", "Admin Dashboard"])
        st.divider()
        st.info("💡 **Voice Tip:** Click the blue microphone icon at the bottom right to speak!")

    if menu == "Chat Assistant":
        run_chat_assistant(cfg)
    else:
        render_admin_dashboard()


def run_chat_assistant(cfg):
    # Standard Header
    st.title("🏨 Grand Hotel AI Concierge")
    st.caption("Your personal assistant for bookings and hotel services.")

    # File Uploader in Expander (Cleaner UI)
    with st.expander("📂 Admin: Upload Hotel Documents"):
        uploaded_files = st.file_uploader(
            "Upload policies (PDF)",
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
    # Container to keep messages scrollable
    chat_container = st.container(height=500)

    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Hi! Ask me about room prices, amenities, or say 'I want to book a room'.")
            
        for msg in st.session_state.messages:
            # Reverted to standard visible avatars
            with st.chat_message(msg["role"], avatar=BOT_AVATAR if msg["role"]=="assistant" else USER_AVATAR):
                st.write(msg["content"])

    # --- INPUTS ---
    user_input = None
    input_source = "text"
    
    # 1. Voice Input (Floating Bottom Right)
    audio_val = st.audio_input("Voice", label_visibility="collapsed")
    
    # 2. Text Input (Standard Bottom)
    text_val = st.chat_input("Type your message...")

    # Logic
    if audio_val:
        with st.spinner("🎧 Transcribing..."):
            transcribed = transcribe_audio(audio_val)
            if transcribed:
                user_input = transcribed
                input_source = "audio"
    
    if text_val:
        user_input = text_val
        input_source = "text"

    if not user_input:
        return

    # Update UI
    with chat_container:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(user_input)
    
    store_message(st.session_state.messages, "user", user_input)

    # --- INTENT & ROUTING ---
    detected_intent = detect_intent(user_input)
    final_intent = detected_intent
    
    # Removed generic words like "room" to prevent booking confusion
    rag_keywords = [ "suite room", "deluxe", 
        "price", "cost", "rate", "wifi", "pool", "gym", "spa", "parking",
        "check-in", "check-out", "policy", "refund", 
        "breakfast", "food", "restaurant", "location", "near", "dinner","lunch",
        "service","services" ,"amenit","aminities", "offer", "facility", "facilities", "type"
    ]
    
    check_booking_keywords = ["check booking", "status", "my booking"]
    # Strong keywords to trigger booking immediately
    start_booking_keywords = ["book", "yes", "sure", "yep", "confirm", "reservation", "need"]
    
    lower_input = user_input.lower()

    # 1. Check for Cancellation (Priority)
    if "cancel" in lower_input:
        final_intent = "booking"

    # 2. Check for Active Booking
    elif st.session_state.booking_state.active:
        # If user asks a specific question (e.g. "What is the price?") let RAG answer
        # But ensure it's not a booking answer (like "deluxe room")
        if any(kw in lower_input for kw in rag_keywords) or "?" in user_input:
             final_intent = "faq_rag"
        else:
             final_intent = "booking"

    # 3. Check for New Booking Intent (Priority over RAG)
    elif any(kw in lower_input for kw in start_booking_keywords) or detected_intent == "booking":
        final_intent = "booking"

    # 4. Check for Booking Status
    elif any(kw in lower_input for kw in check_booking_keywords):
        final_intent = "check_booking"
        
    # 5. Fallback to RAG for questions
    elif any(kw in lower_input for kw in rag_keywords) or "?" in user_input:
        final_intent = "faq_rag"

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
        # Final Fallback Logic
        if "?" in user_input:
             response_text = handle_faq_intent(user_input)
        elif "thank" in lower_input:
             response_text = "You're welcome! Let me know if you need anything else."
        elif any(w in lower_input.split() for w in ["okay", "ok", "alright", "sure", "cool"]):
             response_text = "Great! Let me know if there's anything else I can help you with."
        else:
             response_text = "I'm not sure I understood. Would you like to make a booking?"

    # Display Assistant Response
    store_message(st.session_state.messages, "assistant", response_text)
    with chat_container:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.write(response_text)
            
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
        return "🚫 Booking cancelled."

    # --- CONFIRMATION HANDLING ---
    if state.awaiting_confirmation:
        # 1. CHECK CONFIRMATION
        if "confirm" in lower_msg or lower_msg in ("yes", "yes, confirm"):
            payload = state.to_payload()
            result = booking_persistence_tool(cfg, payload)

            if not result["success"]:
                st.session_state.booking_state = BookingState()
                return f"⚠️ Error: {result['error']}"

            booking_id = result["booking_id"]
            email_body = (
                f"Your booking is confirmed ! 🎉\n\n"
                f"Booking ID: {booking_id}\n\n"
                f"{generate_confirmation_text(state)}"
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

        # 2. ALLOW CORRECTIONS (Explicit Fallthrough)
        pass 

    # --- PROCESS UPDATES (Normal flow or Correction flow) ---
    state = update_state_from_message(user_message, state)
    st.session_state.booking_state = state

    if state.errors:
        field, msg = next(iter(state.errors.items()))
        return f"⚠️ {msg}"

    missing = get_missing_fields(state)

    if missing:
        # --- WELCOME LOGIC ---
        question = next_question_for_missing_field(missing[0])
        gratitude_words = ["thank", "thanks", "thx", "appreciate"]
        if any(w in lower_msg for w in gratitude_words):
            return f"You're welcome! {question}"
        return question

    summary = generate_confirmation_text(state)
    state.awaiting_confirmation = True
    st.session_state.booking_state = state

    # If this was a correction, we acknowledge the updated details
    if state.awaiting_confirmation and "confirm" not in lower_msg:
         return f"**Updated details:**\n\n{summary}\n\nType **'confirm'** if this looks correct."

    return f"**Please confirm details:**\n\n{summary}\n\nType **'confirm'**."


def handle_faq_intent(user_message: str) -> str:
    store: RAGStore = st.session_state.rag_store
    if store is None or store.size == 0:
        return "I can't answer that yet. Please upload hotel documents first."
    else:
        # --- CHANGED: Pass chat history for context ---
        return rag_tool(store, user_message, st.session_state.messages)


if __name__ == "__main__":
    main()