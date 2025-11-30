🏨 AI Hotel Booking Assistant

A sophisticated AI-powered chatbot for hotel bookings and guest services. This assistant handles room reservations, answers questions about hotel amenities using PDF documents (RAG), and provides a seamless conversational experience with voice support.

✨ Features

- Smart Booking Flow: Conversational booking process that collects Name, Email, Phone, Room Type, Date, and Time.

- RAG (Retrieval-Augmented Generation): Upload hotel policy PDFs to let the bot answer specific questions about services, rules, and amenities.

- Voice Interaction: Speak to the assistant and hear responses read aloud (Speech-to-Text & Text-to-Speech).

- Database Integration: Stores all bookings and customer details in Supabase.

- Email Confirmation: Sends an automated confirmation email upon successful booking.

- Admin Dashboard: View, filter, and export booking data (CSV).

- Error Handling: Validates inputs (dates, emails, phone numbers) and handles API errors gracefully.

🛠️ Tech Stack

- Frontend: Streamlit

- LLM & Embeddings: Google Gemini (via google-generativeai)

- Database: Supabase (PostgreSQL)

- Vector Store: FAISS (Local CPU version)

- Audio: gTTS (Google Text-to-Speech)

- Language: Python 3.9+

🚀 Setup & Installation for deploying locally:

1. Clone the Repository

git clone https://github.com/Aishwaryatavidisetty/My-hotel-assistant

cd My-hotel-assistant


2. Create a Virtual Environment

python -m venv venv
#### Windows:
venv\Scripts\activate
#### Mac/Linux:
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Configure Secrets

Create a .streamlit/secrets.toml file in the root directory:

#### .streamlit/secrets.toml

[google]

api_key = "YOUR_GEMINI_API_KEY"

[supabase]

url = "YOUR_SUPABASE_URL"

service_key = "YOUR_SUPABASE_SERVICE_ROLE_KEY"

[email]

smtp_host = "smtp.gmail.com"

smtp_port = 587

smtp_user = "your-email@gmail.com"

smtp_password = "your-app-password"  # Generate this in Google Account > Security

from_email = "your-email@gmail.com"

from_name = "AI Hotel Booking Assistant"


5. Run the Application

streamlit run app/main.py


🛡️ Database Schema (Supabase)

Run this SQL in your Supabase SQL Editor to set up the tables:

create table public.customers (
  customer_id uuid not null default gen_random_uuid() primary key,
  created_at timestamp with time zone not null default now(),
  name text null,
  email text null,
  phone text null
);

create table public.bookings (
  id uuid not null default gen_random_uuid() primary key,
  created_at timestamp with time zone not null default now(),
  customer_id uuid references public.customers (customer_id),
  booking_type text null,
  date text null,
  time text null,
  status text null
);

alter table public.customers disable row level security;
alter table public.bookings disable row level security;


📝 Usage Guide

Chat Assistant:

- Type or speak to start booking.

- Ask questions like "What is the price of a deluxe room?" (requires PDF upload).

- Use natural language to correct details: "I want to change the date"

Admin Dashboard:

- Navigate via the sidebar.

- View all bookings, filter by status, cancel bookings, or download CSV reports.

🤖 Streamlit Deployement link

https://my-hotel-assistant-upwybbnfxrc4hzqgqozfwf.streamlit.app/

