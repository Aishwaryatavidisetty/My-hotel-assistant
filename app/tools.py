from typing import Dict, Any, List
from email.mime.text import MIMEText
import smtplib
import traceback
from datetime import datetime
import streamlit as st

from config import AppConfig
from db.database import get_supabase_client


# --- BOOKING PERSISTENCE TOOL ----------------------------------------------

def booking_persistence_tool(cfg, booking_payload):
    try:
        supabase = get_supabase_client()

        email = booking_payload["email"]

        # 1. Check if customer exists
        customer_lookup = (
            supabase.table("customers").select("*").eq("email", email).execute()
        )

        # 2. If not, insert customer
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
            if not customer_insert.data:
                raise Exception("Failed to insert customer. No data returned.")
            customer_id = customer_insert.data[0]["customer_id"]
        else:
            customer_id = customer_lookup.data[0]["customer_id"]

        # 3. Insert Booking
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

        if not booking_insert.data:
             raise Exception("Failed to insert booking. No data returned.")

        booking_id = booking_insert.data[0]["id"]

        return {
            "success": True,
            "booking_id": booking_id,
            "customer_id": customer_id,
            "error": None,
        }

    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'message'):
             error_msg = e.message
        elif hasattr(e, 'details'):
             error_msg = e.details
             
        st.error(f"Database Error Details: {error_msg}")
        return {
            "success": False, 
            "booking_id": None, 
            "customer_id": None, 
            "error": error_msg
        }


# --- NEW: BOOKING RETRIEVAL TOOL -------------------------------------------

def find_booking_by_email(email: str) -> List[Dict[str, Any]]:
    """Fetches active bookings for a given email address."""
    try:
        supabase = get_supabase_client()
        
        # 1. Find customer ID by email
        cust_res = supabase.table("customers").select("customer_id, name").eq("email", email).execute()
        
        if not cust_res.data:
            return []
            
        customer = cust_res.data[0]
        cid = customer['customer_id']
        name = customer['name']
        
        # 2. Find bookings for this customer
        # We assume 'id' is the booking ID based on previous schema
        book_res = supabase.table("bookings").select("*").eq("customer_id", cid).execute()
        
        results = []
        for b in book_res.data:
            results.append({
                "booking_id": b.get("id"),
                "customer_name": name,
                "type": b.get("booking_type"),
                "date": b.get("date"),
                "time": b.get("time"),
                "status": b.get("status")
            })
            
        return results

    except Exception as e:
        print(f"Error fetching bookings: {e}")
        return []


# --- EMAIL TOOL -------------------------------------------------------------

def email_tool(cfg: AppConfig, to_email: str, subject: str, body: str) -> Dict[str, Any]:
    if not cfg.email or not cfg.email.smtp_host:
        print("Email tool skipped: No SMTP config provided.")
        return {"success": True, "error": None}

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