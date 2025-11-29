# app/config.py
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
    anon_key: str
    service_key: str


@dataclass
class AppConfig:
    gemini: GeminiConfig
    email: EmailConfig
    supabase: SupabaseConfig


# ---------------------- LOADING ----------------------

def load_config() -> AppConfig:
    secrets = st.secrets

    # --- Gemini ---
    gemini_cfg = GeminiConfig(
        api_key=secrets["gemini"]["api_key"],
    )

    # --- Email ---
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
        anon_key=secrets["supabase"]["anon_key"],
        service_key=secrets["supabase"]["service_key"],
    )

    return AppConfig(
        gemini=gemini_cfg,
        email=email_cfg,
        supabase=supabase_cfg,
    )
