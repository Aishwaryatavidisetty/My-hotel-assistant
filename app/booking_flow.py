from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import Optional, Dict, Any, List
import json
import re

import streamlit as st
from email_validator import validate_email as _validate_email, EmailNotValidError
import google.generativeai as genai


BOOKING_FIELDS = [
    "customer_name",
    "email",
    "phone",
    "booking_type",
    "date",
    "time",
]


@dataclass
class BookingState:
    customer_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    booking_type: Optional[str] = None
    date: Optional[date] = None
    time: Optional[time] = None
    
    active: bool = False 

    awaiting_confirmation: bool = False
    errors: Dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "customer_name": self.customer_name,
            "email": self.email,
            "phone": self.phone,
            "booking_type": self.booking_type,
            "date": self.date,
            "time": self.time,
        }


# ----------------- VALIDATORS ------------------------

def validate_email(email: str) -> bool:
    try:
        _validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False


def parse_date_str(val: str) -> Optional[date]:
    try:
        return datetime.strptime(val.strip(), "%Y-%m-%d").date()
    except:
        return None


def parse_time_str(val: str) -> Optional[time]:
    if not val:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(val.strip(), fmt).time()
        except:
            continue
    return None


# ----------------- SLOT HANDLING ------------------------

def get_missing_fields(state: BookingState) -> List[str]:
    missing = []
    for f in BOOKING_FIELDS:
        if getattr(state, f, None) in (None, ""):
            missing.append(f)
    return missing


def generate_confirmation_text(state: BookingState) -> str:
    # Use Markdown bullet points to force new lines
    return (
        f"- **Name:** {state.customer_name}\n"
        f"- **Email:** {state.email}\n"
        f"- **Phone:** {state.phone or 'N/A'}\n"
        f"- **Room Type:** {state.booking_type}\n"
        f"- **Date:** {state.date}\n"
        f"- **Time:** {state.time}"
    )


# ----------------- GEMINI EXTRACTION ------------------------

def _configure_gemini():
    try:
        if "google" in st.secrets:
            api_key = st.secrets["google"]["api_key"]
        elif "gemini" in st.secrets:
            api_key = st.secrets["gemini"]["api_key"]
        else:
            api_key = st.secrets.get("google_api_key", "")
            
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")

def llm_extract_booking_fields(message: str, state: BookingState) -> Dict[str, Any]:
    _configure_gemini()
    
    missing = get_missing_fields(state)
    expected_field = missing[0] if missing else "none"
    today = date.today().isoformat()

    # --- DYNAMIC CONTEXT ---
    if state.awaiting_confirmation:
        context_str = (
            f"CURRENT CONTEXT: The user is reviewing their booking details. "
            f"They might say 'confirm', or they might provide a CORRECTION. "
            f"If they say 'Change name' WITHOUT providing a new name, return 'RESET' for customer_name. "
            f"Same logic for other fields: return 'RESET' if they want to change it but didn't say what to."
        )
    else:
        context_str = f"CURRENT CONTEXT: The system is asking the user for: '{expected_field}'."

    system_prompt = (
        "You extract booking fields from user text. "
        f"{context_str} "
        f"TODAY'S DATE: {today}. "
        "If the user provides a short answer (e.g. 'John' or 'tomorrow'), assume it refers to the requested field. "
        "Return a valid JSON object (no markdown formatting) with keys: "
        "customer_name, email, phone, booking_type, date, time. "
        "Use date format YYYY-MM-DD and time HH:MM (24-hour). "
        "If a field is missing, set it to null. "
        "CRITICAL: If the user indicates they want to change/correct a field but does NOT provide the new value, set that field to the string 'RESET'. "
        "Do not include ```json ... ``` wrappers, just raw JSON."
    )

    prompt = f"{system_prompt}\n\nUser Message: {message}"

    models_to_try = [
        'gemini-2.0-flash', 
        'gemini-2.0-flash-lite',
        'gemini-flash-latest',
        'gemini-pro-latest'
    ]
    
    content = ""
    last_error = None

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            content = response.text
            break # Success
        except Exception as e:
            last_error = e
            continue
    
    if not content:
        return {}

    try:
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        return {}


# ----------------- STATE UPDATE (IMPROVED VALIDATION + RESET LOGIC) ------------------------

def update_state_from_message(message: str, state: BookingState) -> BookingState:
    state.active = True
    
    # 1. Check what we were looking for BEFORE extracting
    missing_before = get_missing_fields(state)
    target_field = missing_before[0] if missing_before else None

    # 2. Extract
    extracted = llm_extract_booking_fields(message, state)
    state.errors.clear()

    # --- Name ---
    if extracted.get("customer_name"):
        val = extracted["customer_name"].strip()
        if val == "RESET":
            state.customer_name = None
        elif len(val) < 2:
             state.errors["customer_name"] = "Name looks too short. Please provide your full name."
        else:
            state.customer_name = val

    # --- Email ---
    if extracted.get("email"):
        val = extracted["email"].strip()
        if val == "RESET":
            state.email = None
        elif validate_email(val):
            state.email = val
        else:
            state.errors["email"] = "That email looks invalid. Please try format: name@example.com"

    # --- Phone ---
    if extracted.get("phone"):
        val = extracted["phone"].strip()
        if val == "RESET":
            state.phone = None
        else:
            digits = re.sub(r'\D', '', val)
            if len(digits) < 10 or len(digits) > 10:
                 state.errors["phone"] = "Invalid phone number. Please enter a valid mobile number."
            else:
                state.phone = val
    elif target_field == "phone":
        state.errors["phone"] = "Invalid phone number."

    # --- Booking Type ---
    if extracted.get("booking_type"):
        val = extracted["booking_type"].strip()
        if val == "RESET":
            state.booking_type = None
        elif len(val) < 2:
             state.errors["booking_type"] = "Invalid room type. Please specify Standard, Deluxe, or Suite."
        else:
            state.booking_type = val
    elif target_field == "booking_type":
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["option", "type", "available", "what", "which"]):
            state.errors["booking_type"] = (
                "We have the following rooms available:\n\n"
                "- **Standard Room**\n"
                "- **Deluxe Room**\n"
                "- **Executive Suite**\n\n"
                "Which one would you like to book?"
            )

    # --- Date Error Handling ---
    if extracted.get("date"):
        val = extracted["date"]
        if val == "RESET":
            state.date = None
        else:
            parsed = parse_date_str(val)
            if parsed:
                if parsed < date.today():
                    state.errors["date"] = "Invalid date (past). Please choose an upcoming date (YYYY-MM-DD)."
                else:
                    state.date = parsed
            else:
                state.errors["date"] = "Invalid date format. Please use YYYY-MM-DD."
    elif target_field == "date":
        # Specific error if we asked for date but got something else
        state.errors["date"] = "Invalid date. Please enter format YYYY-MM-DD."

    # --- Time Error Handling ---
    if extracted.get("time"):
        val = extracted["time"]
        if val == "RESET":
            state.time = None
        else:
            parsed = parse_time_str(val)
            if parsed:
                state.time = parsed
            else:
                state.errors["time"] = "Invalid time format. Please use HH:MM."
    elif target_field == "time":
        # Specific error if we asked for time but got something else
        state.errors["time"] = "Invalid time. Please use 24-hour format (e.g., 14:00)."

    # --- Reorder Errors (Prioritize Current Question) ---
    if target_field and target_field in state.errors:
        priority_error = {target_field: state.errors[target_field]}
        for k, v in state.errors.items():
            if k != target_field:
                priority_error[k] = v
        state.errors = priority_error

    return state


# ----------------- QUESTIONS ------------------------

def next_question_for_missing_field(field_name: str) -> str:
    prompts = {
        "customer_name": "May I know the guest name?",
        "email": "What's your email address for confirmation?",
        "phone": "Your phone number? ",
        "booking_type": "What type of room would you like to book? (Standard, Deluxe, Suite)",
        "date": "What check-in date? Please use YYYY-MM-DD.",
        "time": "What arrival time? Please use HH:MM (24-hour).",
    }
    return prompts.get(field_name, f"Please provide {field_name}.")