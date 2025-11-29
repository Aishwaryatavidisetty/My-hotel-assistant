# app/tools.py

from typing import Dict, Any
from email.mime.text import MIMEText
import smtplib
import traceback
from datetime import datetime

from config import AppConfig
from db.database import get_supabase_client


# --- BOOKING PERSISTENCE TOOL ----------------------------------------------

def booking_persistence_tool(cfg, booking_payload):
    supabase = get_supabase_client()

    email = booking_payload["email"]

    customer_lookup = (
        supabase.table("customers").select("*").eq("email", email).execute()
    )

    if len(customer_lookup.data) == 0:
        customer_insert = (
            supabase.table("customers")
            .insert(
                {
                    "name": booking_payload["customer_name"],
                    "email": email,
                    "phone": booking_payload.get("phone"),
                }
            )
            .execute()
        )
        customer_id = customer_insert.data[0]["customer_id"]
    else:
        customer_id = customer_lookup.data[0]["customer_id"]

    booking_insert = (
        supabase.table("bookings")
        .insert(
            {
                "customer_id": customer_id,
                "booking_type": booking_payload["booking_type"],
                "date": str(booking_payload["date"]),
                "time": str(booking_payload["time"]),
                "status": "confirmed",
                "created_at": datetime.utcnow().isoformat(),
            }
        )
        .execute()
    )

    booking_id = booking_insert.data[0]["id"]

    return {
        "success": True,
        "booking_id": booking_id,
        "customer_id": customer_id,
        "error": None,
    }


# --- EMAIL TOOL -------------------------------------------------------------

def email_tool(cfg: AppConfig, to_email: str, subject: str, body: str) -> Dict[str, Any]:
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = f"{cfg.email.from_name} <{cfg.email.from_email}>"
    msg["To"] = to_email

    try:
        with smtplib.SMTP(cfg.email.smtp_host, cfg.email.smtp_port) as server:
            server.starttls()
            server.login(cfg.email.smtp_user, cfg.email.smtp_password)
            server.send_message(msg)
        return {"success": True, "error": None}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# --- OPTIONAL WEB SEARCH TOOL ----------------------------------------------

def web_search_tool(query: str) -> Dict[str, Any]:
    return {
        "success": False,
        "results": [],
        "error": "Web search tool not implemented.",
    }
