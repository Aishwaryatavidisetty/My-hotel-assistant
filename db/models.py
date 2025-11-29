# db/models.py
"""
Supabase does not require ORM model classes.
Tables created in your Supabase dashboard:

Table: customers
- customer_id (int, PK)
- name (text)
- email (text, unique)
- phone (text)

Table: bookings
- id (int, PK)
- customer_id (int, FK → customers.customer_id)
- booking_type (text)
- date (date)
- time (text or time)
- status (text)
- created_at (timestamp)
"""
