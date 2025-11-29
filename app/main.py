from __future__ import annotations

import sys
import os

# --- CRITICAL FIX: Add project root to sys.path ---
# This block ensures that Python can find the 'db' folder (which is in the parent directory)
# regardless of how you run the Streamlit command.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st

# IMPORTS (Now safe because sys.path includes the root)
from config import load_config
from chat_logic import detect_intent, store_message
from rag_pipeline import RAGStore, RAGConfig, build_rag_store_from_uploads
from rag_pipeline import rag_tool 
from tools import booking_persistence_tool, email_tool
from admin_dashboard import render_admin_dashboard
# from db.database import init_db # REMOVED: Function does not exist in db.database

from booking_flow import (
    BookingState,
    get_missing_fields,
    generate_confirmation_text,
    update_state_from_message,
    next_question_for_missing_field,
)


def _init_app_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "booking_state" not in st.session_state:
        st.session_state.booking_state = BookingState()
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = None
    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks = []


def main():
    st.set_page_config(
        page_title="AI Hotel Booking Assistant",
        page_icon="🏨",
        layout="wide",
    )

    cfg = load_config()
    # init_db() # REMOVED: Skipping DB initialization in code
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

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("How can I help you with your hotel stay today?")
    if not user_input:
        return

    store_message(st.session_state.messages, "user", user_input)
    intent = detect_intent(user_input)

    if intent == "booking":
        handle_booking_intent(cfg, user_input)
    elif intent == "faq_rag":
        handle_faq_intent(user_input)
    elif intent == "small_talk":
        respond("Hello! I can help you book rooms or answer questions about the hotel.")
    else:
        respond(
            "I’m not sure I understood. "
            "Are you trying to make a hotel booking or asking about hotel details?"
        )


def handle_booking_intent(cfg, user_message: str):
    state: BookingState = st.session_state.booking_state
    lower_msg = user_message.strip().lower()

    # Confirmation
    if state.awaiting_confirmation:
        if "confirm" in lower_msg or lower_msg in ("yes", "yes, confirm"):
            payload = state.to_payload()
            result = booking_persistence_tool(cfg, payload)

            if not result["success"]:
                respond(f"Error saving booking: {result['error']}")
                st.session_state.booking_state = BookingState()
                return

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

            if not email_result["success"]:
                respond(
                    f"Booking confirmed (ID {booking_id}) but email failed: {email_result['error']}"
                )
            else:
                respond(
                    f"🎉 Booking confirmed! ID: {booking_id}. "
                    "A confirmation email has been sent."
                )

            st.session_state.booking_state = BookingState()
            return

        if "cancel" in lower_msg:
            respond("Booking cancelled. Let me know if you'd like to start again.")
            st.session_state.booking_state = BookingState()
            return

        respond("Type 'confirm' or 'cancel'.")
        return

    # Update state
    state = update_state_from_message(user_message, state)
    st.session_state.booking_state = state

    if state.errors:
        field, msg = next(iter(state.errors.items()))
        respond(msg)
        return

    missing = get_missing_fields(state)

    if missing:
        next_field = missing[0]
        respond(next_question_for_missing_field(next_field))
        return

    # Ask for confirmation
    summary = generate_confirmation_text(state)
    state.awaiting_confirmation = True
    st.session_state.booking_state = state

    respond(
        "Here are your booking details:\n\n"
        f"{summary}\n"
        "Type **'confirm'** to finalize or **'cancel'**."
    )


def handle_faq_intent(user_message: str):
    store: RAGStore = st.session_state.rag_store

    if store is None or store.size == 0:
        respond(
            "No hotel documents indexed yet. Upload PDFs and click "
            "'Build Knowledge Base', then ask your question again."
        )
    else:
        # Calls the function imported from rag_pipeline.py
        respond(rag_tool(store, user_message))


def respond(text: str):
    store_message(st.session_state.messages, "assistant", text)
    with st.chat_message("assistant"):
        st.write(text)


if __name__ == "__main__":
    main()