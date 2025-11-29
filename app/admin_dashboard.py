# app/admin_dashboard.py

import streamlit as st
from db.database import get_supabase_client


def render_admin_dashboard():
    st.title("📊 Admin Dashboard – Hotel Bookings")

    supabase = get_supabase_client()

    # --- Filters ---
    st.subheader("Filters")
    name_filter = st.text_input("Filter by guest name:")
    email_filter = st.text_input("Filter by email:")
    date_filter = st.date_input("Filter by booking date:", value=None)

    # --- Load customers & bookings ---
    customers_resp = supabase.table("customers").select("*").execute()
    bookings_resp = supabase.table("bookings").select("*").execute()

    customers = customers_resp.data or []
    bookings = bookings_resp.data or []

    # dict for O(1) lookup
    customer_map = {c["customer_id"]: c for c in customers}

    # --- Combine rows ---
    rows = []
    for booking in bookings:
        cust = customer_map.get(booking["customer_id"], {})
        rows.append({
            "Booking ID": booking.get("id"),
            "Guest Name": cust.get("name"),
            "Email": cust.get("email"),
            "Phone": cust.get("phone"),
            "Booking Type": booking.get("booking_type"),
            "Date": booking.get("date"),
            "Time": booking.get("time"),
            "Status": booking.get("status"),
            "Created At": booking.get("created_at"),
        })

    # --- Apply Filters ---
    if name_filter:
        rows = [r for r in rows if name_filter.lower() in (r["Guest Name"] or "").lower()]

    if email_filter:
        rows = [r for r in rows if email_filter.lower() in (r["Email"] or "").lower()]

    if date_filter:
        rows = [r for r in rows if r["Date"] == str(date_filter)]

    # --- Display ---
    if len(rows) == 0:
        st.info("No bookings match your filters.")
    else:
        st.subheader("Results")
        st.dataframe(rows, use_container_width=True)
