import streamlit as st
import pandas as pd
import plotly.express as px
from db.database import get_supabase_client

def render_admin_dashboard():
    st.title("📊 Hotel Admin Dashboard")

    # --- Simple Password Protection (Optional) ---
    password = st.sidebar.text_input("Admin Password", type="password")
    if password != "admin123":  # You can change this or use secrets
        st.warning("Please enter the correct admin password to view data.")
        return

    supabase = get_supabase_client()

    # --- Fetch Data ---
    try:
        # Join bookings with customers
        # Note: Supabase-py join syntax can be tricky, fetching separately is safer for simple dashboards
        bookings_response = supabase.table("bookings").select("*").execute()
        customers_response = supabase.table("customers").select("*").execute()
        
        if not bookings_response.data:
            st.info("No bookings found in the database.")
            return

        bookings_df = pd.DataFrame(bookings_response.data)
        customers_df = pd.DataFrame(customers_response.data)

        # Merge dataframes on customer_id
        # Ensure ID columns are matching types
        if not bookings_df.empty and not customers_df.empty:
            df = pd.merge(
                bookings_df, 
                customers_df, 
                left_on="customer_id", 
                right_on="customer_id", 
                how="left"
            )
        else:
            df = bookings_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # --- KPI Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Bookings", len(df))
    confirmed = len(df[df['status'] == 'confirmed']) if 'status' in df.columns else 0
    col2.metric("Confirmed", confirmed)
    cancelled = len(df[df['status'] == 'cancelled']) if 'status' in df.columns else 0
    col3.metric("Cancelled", cancelled)

    # --- Filters ---
    st.divider()
    st.subheader("Booking Management")
    
    status_filter = st.multiselect(
        "Filter by Status", 
        options=df["status"].unique() if "status" in df.columns else [],
        default=df["status"].unique() if "status" in df.columns else []
    )
    
    if status_filter:
        filtered_df = df[df["status"].isin(status_filter)]
    else:
        filtered_df = df

    # --- Main Data Table ---
    # Reorder columns for readability
    display_cols = [
        "name", "email", "phone", "booking_type", "date", "time", "status", "id"
    ]
    # Filter columns that actually exist
    final_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(filtered_df[final_cols], use_container_width=True)

    # --- Actions: Cancel Booking ---
    st.write("### Actions")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        booking_id_to_cancel = st.text_input("Enter Booking ID to Cancel")
        if st.button("Cancel Booking"):
            if booking_id_to_cancel:
                try:
                    supabase.table("bookings").update({"status": "cancelled"}).eq("id", booking_id_to_cancel).execute()
                    st.success(f"Booking {booking_id_to_cancel} cancelled!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to cancel: {e}")

    # --- Actions: Export ---
    with c2:
        st.write("### Export")
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download as CSV",
            csv,
            "hotel_bookings.csv",
            "text/csv",
            key='download-csv'
        )