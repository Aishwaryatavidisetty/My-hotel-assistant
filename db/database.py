# db/database.py

from supabase import create_client, Client
import streamlit as st


def get_supabase_client() -> Client:
    """
    Returns a cached Supabase client.
    Uses service_role_key because bookings and customer creation
    require RLS bypass for inserts.
    """

    if "supabase_client" not in st.session_state:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_key"]  # service key for write access
        st.session_state.supabase_client = create_client(url, key)

    return st.session_state.supabase_client
