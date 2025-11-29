from __future__ import annotations

from dataclasses import dataclass
import streamlit as st


# ---------------------- DATA CLASSES ----------------------

@dataclass
class GeminiConfig:
    api_key: str


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    from_email: str
    from_name: str


@dataclass
class SupabaseConfig:
    url: str
    service_key: str  # Removed anon_key as we primarily use service_key for admin tasks


@dataclass
class AppConfig:
    gemini: GeminiConfig
    email: EmailConfig
    supabase: SupabaseConfig


# ---------------------- LOADING ----------------------

def load_config() -> AppConfig:
    secrets = st.secrets

    # --- Gemini (Google) ---
    # Checks for [google] section first, then falls back to [gemini]
    if "google" in secrets:
        api_key = secrets["google"]["api_key"]
    elif "gemini" in secrets:
        api_key = secrets["gemini"]["api_key"]
    else:
        # Fallback for simple key entry
        api_key = secrets.get("google_api_key", "")

    gemini_cfg = GeminiConfig(api_key=api_key)

    # --- Email ---
    # users often store ports as integers or strings, converting to int ensures safety
    email_cfg = EmailConfig(
        smtp_host=secrets["email"]["smtp_host"],
        smtp_port=int(secrets["email"]["smtp_port"]),
        smtp_user=secrets["email"]["smtp_user"],
        smtp_password=secrets["email"]["smtp_password"],
        from_email=secrets["email"]["from_email"],
        from_name=secrets["email"]["from_name"],
    )

    # --- Supabase ---
    supabase_cfg = SupabaseConfig(
        url=secrets["supabase"]["url"],
        service_key=secrets["supabase"]["service_key"],
    )

    return AppConfig(
        gemini=gemini_cfg,
        email=email_cfg,
        supabase=supabase_cfg,
    )