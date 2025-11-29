# app/chat_logic.py

from __future__ import annotations
from typing import Literal, Dict, Any, List

Intent = Literal["booking", "faq_rag", "small_talk", "unknown"]


def detect_intent(user_message: str) -> Intent:
    text = user_message.lower()

    booking_keywords = [
        "book", "booking", "reserve", "reservation",
        "room", "suite", "stay", "check in", "check-in", "checkin",
        "night", "nights", "need a room", "want a room",
        "looking for a room",
    ]

    faq_keywords = [
        "price", "rate", "cost", "wifi", "internet",
        "pool", "parking", "breakfast", "amenities",
        "check-out", "checkout",
        "pet", "policy", "refund", "cancellation",
        "details", "information", "tell me about",
        "do you have", "availability",
    ]

    smalltalk_keywords = [
        "hi", "hello", "hey", "good morning", "good evening",
        "how are you", "what's up", "how is it going",
    ]

    # Precedence logic
    if any(k in text for k in booking_keywords):
        return "booking"
    if any(k in text for k in faq_keywords):
        return "faq_rag"
    if any(k in text for k in smalltalk_keywords):
        return "small_talk"

    return "unknown"


def store_message(history: List[Dict[str, Any]], role: str, content: str, max_messages: int = 25) -> None:
    history.append({"role": role, "content": content})
    if len(history) > max_messages:
        del history[: len(history) - max_messages]


def last_user_message(history: List[Dict[str, Any]]) -> str:
    for msg in reversed(history):
        if msg["role"] == "user":
            return msg["content"]
    return ""
