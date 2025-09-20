"""
Interactive SDR Simulator with real Email send/receive capability.
Now with auto-insert lead fix (unknown senders and new overrides get added).
Modified with static email settings and combined generate/send logic.
"""

import os
import random
import time
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# email libraries
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.header import decode_header
from email.utils import parseaddr

# ---------------------------
# Optional Cohere
# ---------------------------
try:
    import cohere
    COHERE_INSTALLED = True
except Exception:
    COHERE_INSTALLED = False

# ---------------------------
# CONFIG
# ---------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
PRIMARY_MODEL = "command-r7b-12-2024"
FALLBACK_MODEL = "command"
PERSONA_INSTRUCTION = "You are friendly, calm, concise and professional. Keep it warm and helpful."

# STATIC EMAIL SETTINGS - Cannot be changed
STATIC_EMAIL_SETTINGS = {
    "from_email": "crabstertech@gmail.com",
    "password": "tmur jmje bybl kzql",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 465,
    "imap_server": "imap.gmail.com",
    "imap_port": 993,
    "use_ssl": True,
}

# ---------------------------
# DATABASE (logs) with corruption recovery
# ---------------------------
DB_PATH = "negotiation.db"
BACKUP_DB_PATH = "negotiation_backup.db"

def create_fresh_database(db_path):
    """Create a fresh database with proper schema"""
    if os.path.exists(db_path):
        # Create backup before removing
        backup_name = f"corrupted_db_backup_{int(time.time())}.db"
        try:
            os.rename(db_path, backup_name)
            st.warning(f"Corrupted database backed up as {backup_name}")
        except:
            os.remove(db_path)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE logs(
            ts REAL,
            lead_name TEXT,
            channel TEXT,
            agent_message TEXT,
            human_reply TEXT,
            classification TEXT,
            reward REAL,
            deal_closed INTEGER,
            deal_quality REAL
        )"""
    )
    
    # Create leads table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS leads(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE
        )"""
    )
    
    conn.commit()
    return conn, c

# Try to connect to existing database
try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    
    # Test if database is working
    c.execute("PRAGMA table_info(logs)")
    cols = c.fetchall()
    migration_performed = False
    
    expected_col_names = [
        "ts",
        "lead_name", 
        "channel",
        "agent_message",
        "human_reply",
        "classification",
        "reward",
        "deal_closed",
        "deal_quality",
    ]
    
    if not cols:
        # Create tables if they don't exist
        c.execute(
            """CREATE TABLE logs(
                ts REAL,
                lead_name TEXT,
                channel TEXT,
                agent_message TEXT,
                human_reply TEXT,
                classification TEXT,
                reward REAL,
                deal_closed INTEGER,
                deal_quality REAL
            )"""
        )
        
        # Create leads table
        c.execute(
            """CREATE TABLE IF NOT EXISTS leads(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE
            )"""
        )
        conn.commit()
    else:
        existing_col_names = [col[1] for col in cols]
        if existing_col_names != expected_col_names:
            migration_performed = True
            c.execute("ALTER TABLE logs RENAME TO logs_backup")
            c.execute(
                """CREATE TABLE logs(
                    ts REAL,
                    lead_name TEXT,
                    channel TEXT,
                    agent_message TEXT,
                    human_reply TEXT,
                    classification TEXT,
                    reward REAL,
                    deal_closed INTEGER,
                    deal_quality REAL
                )"""
            )
            common_cols = [col for col in expected_col_names if col in existing_col_names]
            if common_cols:
                col_str = ",".join(common_cols)
                c.execute(f"INSERT INTO logs ({col_str}) SELECT {col_str} FROM logs_backup")
            conn.commit()

except sqlite3.DatabaseError as e:
    st.error(f"Database corrupted: {e}")
    st.info("Creating fresh database...")
    conn, c = create_fresh_database(DB_PATH)
    migration_performed = False
    st.success("✅ Fresh database created successfully!")

except Exception as e:
    st.error(f"Database error: {e}")
    st.info("Creating fresh database...")
    conn, c = create_fresh_database(DB_PATH)
    migration_performed = False
    st.success("✅ Fresh database created successfully!")

# ---------------------------
# LEADS CSV
# ---------------------------
SAMPLE_CSV = "leads.csv"

def create_empty_leads_csv():
    """Create empty leads CSV with headers only"""
    empty_df = pd.DataFrame(columns=["lead_name", "lead_role", "channel", "email", "previous_messages", "response_probability"])
    empty_df.to_csv(SAMPLE_CSV, index=False)
    return []

def load_leads():
    """Load leads from CSV - only from file, no defaults"""
    try:
        if not os.path.exists(SAMPLE_CSV):
            # Create empty CSV if it doesn't exist
            st.info("leads.csv not found. Creating empty CSV file.")
            return create_empty_leads_csv()
        
        # Try to read existing CSV
        leads_df = pd.read_csv(SAMPLE_CSV)
        
        # If CSV is empty, return empty list
        if leads_df.empty:
            st.info("leads.csv is empty. Use 'Add New Lead' to add leads.")
            return []
        
        # Ensure required columns exist
        required_cols = ["lead_name", "lead_role", "channel", "email", "previous_messages", "response_probability"]
        for col in required_cols:
            if col not in leads_df.columns:
                leads_df[col] = "" if col != "response_probability" else 0.5
        
        # Filter out leads with empty names
        valid_leads = leads_df[leads_df['lead_name'].str.strip() != ''].to_dict(orient="records")
        
        if not valid_leads:
            st.info("No valid leads found in CSV. Use 'Add New Lead' to add leads.")
            return []
        
        return valid_leads
        
    except Exception as e:
        st.error(f"Error loading leads.csv: {e}")
        st.info("Creating new empty CSV file.")
        return create_empty_leads_csv()

def add_lead_to_csv(lead_name, lead_role, channel, email, previous_messages="", response_probability=0.5):
    """Add a new lead to the CSV file"""
    try:
        # Load existing CSV or create empty one
        if os.path.exists(SAMPLE_CSV):
            df = pd.read_csv(SAMPLE_CSV)
        else:
            df = pd.DataFrame(columns=["lead_name", "lead_role", "channel", "email", "previous_messages", "response_probability"])
        
        # Check if lead already exists
        if not df.empty and email in df['email'].values:
            return False, f"Lead with email {email} already exists"
        
        # Add new lead
        new_row = {
            "lead_name": lead_name,
            "lead_role": lead_role, 
            "channel": channel,
            "email": email,
            "previous_messages": previous_messages,
            "response_probability": response_probability
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(SAMPLE_CSV, index=False)
        
        return True, f"Lead {lead_name} added successfully"
        
    except Exception as e:
        return False, f"Error adding lead: {e}"

# Load leads from CSV only
leads = load_leads()

# Make leads globally accessible
globals()['leads'] = leads

# ---------------------------
# DB utility: ensure lead exists (only for database, not CSV)
# ---------------------------
def ensure_lead_exists_in_db_only(conn, lead_name, lead_email):
    """Only ensure lead exists in database, not in CSV (CSV is managed separately)"""
    if not lead_email:
        return
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM leads WHERE email = ?", (lead_email,))
        row = cursor.fetchone()
        exists = row[0] if row else 0
        if not exists:
            cursor.execute(
                "INSERT OR IGNORE INTO leads (name, email) VALUES (?, ?)",
                (lead_name, lead_email)
            )
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error in ensure_lead_exists: {e}")
    except Exception as e:
        st.warning(f"Error ensuring lead exists: {e}")

def check_if_lead_exists_in_csv(email):
    """Check if lead already exists in CSV"""
    try:
        if os.path.exists(SAMPLE_CSV):
            df = pd.read_csv(SAMPLE_CSV)
            if not df.empty and email in df['email'].values:
                return True
        return False
    except:
        return False

# ---------------------------
# SIMPLE ENV
# ---------------------------
class SimpleEnv:
    def __init__(self, leads_list: List[Dict]):
        self.leads = leads_list
        self.current = None
        if self.leads:
            self.reset()

    def reset(self):
        if self.leads:
            self.current = random.choice(self.leads)
            return self.current
        return None

    def get_lead_name(self):
        if self.current:
            return self.current.get("lead_name", "Lead")
        return "No Lead Selected"

    def get_current_lead(self):
        return self.current

    def update_leads(self, new_leads):
        """Update the leads list"""
        self.leads = new_leads
        if self.leads and not self.current:
            self.reset()

env = SimpleEnv(leads)

# ---------------------------
# MESSAGE GENERATOR
# ---------------------------
def generate_agent_message(lead_name: str, channel: str, topic: str = "a quick intro") -> str:
    prompt = (
    f"{PERSONA_INSTRUCTION}\n\n"
    f"Write a short, friendly, and professional {channel} outreach message"
    f"to {lead_name} about {topic}. "
    f"Do not include placeholders or square brackets — fill in details naturally. "
    f"End the message with my name, Sasitharan, as the sender. "
    f"Make the message warm, approachable, and specific to the context."
    )
    if COHERE_INSTALLED and COHERE_API_KEY:
        try:
            client = cohere.Client(COHERE_API_KEY)
            resp = client.chat(model=PRIMARY_MODEL, message=prompt, temperature=0.6)
            if resp.text.strip():
                return resp.text.strip()
        except Exception:
            pass
    if channel.lower() == "email":
        return f"Hi {lead_name},\n\nI hope you're doing well. I wanted to share a quick thought about {topic} — would you be open to a short call?"
    else:
        return f"Hi {lead_name}, hope you're well — I had a quick idea on {topic}. Would you be open to a short chat?"

# ---------------------------
# NEGOTIATOR MESSAGE GENERATOR
# ---------------------------
def generate_negotiator_message(lead_name: str, objection_text: str) -> str:
    prompt = (
        f"You are a skilled sales negotiator. "
        f"Lead {lead_name} replied with an objection: \"{objection_text}\". "
        f"Craft a short, professional, persuasive reply (2-3 sentences). "
        f"Handle the objection positively and keep the door open."
    )
    if COHERE_INSTALLED and COHERE_API_KEY:
        try:
            client = cohere.Client(COHERE_API_KEY)
            resp = client.chat(model=PRIMARY_MODEL, message=prompt, temperature=0.7)
            if resp.text.strip():
                return resp.text.strip()
        except Exception:
            pass
    return f"Hi {lead_name}, I completely understand your concern. Many of our clients felt the same initially, but they found the value far outweighed the cost. Would you be open to a quick call to explore options?"

# ---------------------------
# REPLY CLASSIFIER
# ---------------------------
def classify_reply(text: str) -> str:
    if not text or text.strip() == "":
        return "ignore"
    t = text.lower()
    if any(kw in t for kw in ["book", "schedule", "call", "meeting", "yes", "ok", "tomorrow", "next week"]):
        return "book"
    if any(kw in t for kw in ["price", "cost", "budget", "expensive", "concern", "later"]):
        return "objection"
    if any(kw in t for kw in ["no", "not interested", "don't", "no thanks", "uninterested"]):
        return "reject"
    if any(kw in t for kw in ["interested", "tell me more", "more info", "details", "keen"]):
        return "interested"
    return "ignore"

# ---------------------------
# DEAL QUALITY
# ---------------------------
def compute_deal_quality(lead_record: Dict, classification: str) -> float:
    role = str(lead_record.get("lead_role", "")).lower()
    role_weight = 0.5 if any(k in role for k in ["cto", "founder", "ceo"]) else 0.3
    prior_msg = str(lead_record.get("previous_messages", "") or "")
    prior = 0.2 if prior_msg.strip() else 0.0
    class_bonus = {"book": 0.5, "interested": 0.3, "objection": 0.15, "ignore": 0.0, "reject": -0.2}.get(classification, 0.0)
    return max(0.0, min(1.0, role_weight + prior + class_bonus))

# ---------------------------
# EMAIL HELPERS (SMTP & IMAP)
# ---------------------------
def decode_mime_words(s: Optional[str]) -> str:
    if not s:
        return ""
    parts = decode_header(s)
    decoded = ""
    for part, enc in parts:
        if isinstance(part, bytes):
            try:
                decoded += part.decode(enc or "utf-8", errors="replace")
            except Exception:
                decoded += part.decode("utf-8", errors="replace")
        else:
            decoded += part
    return decoded

def extract_plain_text(msg: email.message.Message) -> str:
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disp = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disp:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        text += payload.decode(charset, errors="replace")
                    except Exception:
                        text += payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except Exception:
                text = payload.decode("utf-8", errors="replace")
    return text

def send_email_smtp(
    smtp_server: str,
    smtp_port: int,
    use_ssl: bool,
    from_email: str,
    password: str,
    to_email: str,
    subject: str,
    body: str,
) -> Tuple[bool, str]:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
            server.login(from_email, password)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()
            server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        return True, "Sent"
    except Exception as e:
        return False, str(e)

def fetch_unseen_imap(
    imap_server: str,
    imap_port: int,
    from_email: str,
    password: str,
    mark_seen: bool = True,
) -> List[Dict]:
    replies = []
    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(from_email, password)
        mail.select("INBOX")
        status, data = mail.search(None, "UNSEEN")
        if status != "OK":
            mail.logout()
            return replies
        email_ids = data[0].split()
        for e_id in email_ids:
            try:
                status, msg_data = mail.fetch(e_id, "(RFC822)")
                if status != "OK":
                    continue
                raw_msg = msg_data[0][1]
                msg = email.message_from_bytes(raw_msg)
                subject = decode_mime_words(msg.get("Subject", ""))
                sender = parseaddr(msg.get("From", ""))[1]
                date = msg.get("Date", "")
                body = extract_plain_text(msg)
                replies.append({
                    "id": e_id,
                    "from": sender,
                    "subject": subject,
                    "body": body,
                    "date": date,
                })
                if mark_seen:
                    mail.store(e_id, "+FLAGS", "\\Seen")
            except Exception:
                continue
        mail.logout()
    except Exception:
        pass
    return replies

# ---------------------------
# Map reply -> lead
# ---------------------------
def map_reply_to_lead(reply: Dict, leads_list: List[Dict]) -> Optional[Dict]:
    sender = (reply.get("from") or "").lower()
    subject = (reply.get("subject") or "").lower()
    for lead in leads_list:
        lead_email = str(lead.get("email", "")).lower()
        if lead_email and lead_email == sender:
            return lead
    for lead in leads_list:
        name = str(lead.get("lead_name", "")).lower()
        if name and name in subject:
            return lead
    return None

# ---------------------------
# Auto-refresh inbox function
# ---------------------------
def auto_check_inbox():
    """Function to automatically check inbox and process replies"""
    global leads  # Make leads accessible in this function
    
    ses = STATIC_EMAIL_SETTINGS
    replies = fetch_unseen_imap(
        ses["imap_server"], 
        int(ses["imap_port"]), 
        ses["from_email"], 
        ses["password"], 
        mark_seen=True
    )
    
    if replies:
        st.success(f"🔄 Auto-refresh: Found {len(replies)} new message(s)")
        processed = 0
        for rep in replies:
            lead = map_reply_to_lead(rep, leads)
            if not lead:
                sender_email = rep.get("from")
                sender_name = sender_email.split('@')[0] if sender_email else "Unknown"
                
                # Only add to CSV if it doesn't already exist
                if not check_if_lead_exists_in_csv(sender_email):
                    success, msg = add_lead_to_csv(
                        sender_name, 
                        "", 
                        "Email", 
                        sender_email, 
                        "Auto-added from reply", 
                        0.5
                    )
                    if success:
                        st.info(f"📧 Auto-added new lead from reply: {sender_name}")
                        # Reload leads after adding new one
                        leads = load_leads()
                        globals()['leads'] = leads
                        lead = map_reply_to_lead(rep, leads)

            if lead:
                classification = classify_reply(rep.get("body", ""))
                reward_map = {"book": 3.0, "interested": 1.0, "objection": 0.7, "ignore": -0.5, "reject": -1.0}
                reward = reward_map.get(classification, -0.5)
                deal_quality = compute_deal_quality(lead, classification)
                deal_closed = 1 if classification == "book" else 0
                agent_message = ""
                try:
                    q = c.execute(
                        "SELECT agent_message FROM logs WHERE lead_name=? AND channel='Email' AND agent_message<>'' ORDER BY ts DESC LIMIT 1",
                        (lead.get("lead_name"),),
                    )
                    row = q.fetchone()
                    if row:
                        agent_message = row[0]
                except Exception:
                    agent_message = ""
                ts = time.time()
                c.execute(
                    "INSERT INTO logs (ts, lead_name, channel, agent_message, human_reply, classification, reward, deal_closed, deal_quality) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ts, lead.get("lead_name"), "Email", agent_message, rep.get("body", ""), classification, float(reward), int(deal_closed), float(deal_quality)),
                )
                conn.commit()
                processed += 1
                
                # Show the processed reply
                with st.expander(f"📧 New Reply from {lead.get('lead_name')} - {classification.upper()}"):
                    st.markdown(f"**From:** {rep.get('from')}  \n**Subject:** {rep.get('subject')}")
                    st.write(rep.get("body")[:1000])
        
        return processed
    return 0

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Interactive SDR Simulator — Email Enabled", layout="wide")
st.title("🤖 Interactive SDR Simulator — Send real emails & poll replies")

if "pending_reply" not in st.session_state:
    st.session_state["pending_reply"] = ""

if "last_inbox_check" not in st.session_state:
    st.session_state["last_inbox_check"] = 0

# Initialize auto-refresh
if "auto_refresh_enabled" not in st.session_state:
    st.session_state["auto_refresh_enabled"] = True

# Auto-refresh inbox every 10 seconds
current_time = time.time()
if (current_time - st.session_state["last_inbox_check"]) >= 10 and st.session_state.get("auto_refresh_enabled", True):
    st.session_state["last_inbox_check"] = current_time
    auto_check_inbox()
    st.rerun()

# Sidebar - Show static email settings (read-only)
with st.sidebar:
    st.header("System")
    if migration_performed:
        st.warning("Database migration performed.")
    else:
        st.info("DB schema OK.")
    st.markdown("---")
    
    st.header("📧 Email Settings (Static)")
    st.info("Email settings are fixed and cannot be changed")
    
    # Display static settings as read-only
    st.text_input("From email", value=STATIC_EMAIL_SETTINGS["from_email"], disabled=True)
    st.text_input("Email password", value="*" * len(STATIC_EMAIL_SETTINGS["password"]), disabled=True, type="password")
    st.text_input("SMTP server", value=STATIC_EMAIL_SETTINGS["smtp_server"], disabled=True)
    st.number_input("SMTP port", value=STATIC_EMAIL_SETTINGS["smtp_port"], disabled=True)
    st.text_input("IMAP server", value=STATIC_EMAIL_SETTINGS["imap_server"], disabled=True)
    st.number_input("IMAP port", value=STATIC_EMAIL_SETTINGS["imap_port"], disabled=True)
    st.checkbox("SMTP use SSL", value=STATIC_EMAIL_SETTINGS["use_ssl"], disabled=True)
    
    st.markdown("---")
    st.header("🔄 Auto-refresh")
    st.session_state["auto_refresh_enabled"] = st.checkbox("Enable auto inbox check (10s)", value=st.session_state.get("auto_refresh_enabled", True))
    
    if st.session_state.get("auto_refresh_enabled"):
        next_check = 10 - (current_time - st.session_state["last_inbox_check"])
        if next_check > 0:
            st.info(f"Next check in: {next_check:.1f}s")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("📋 Lead Management")
    
    # Add new lead section
    with st.expander("➕ Add New Lead", expanded=not leads):
        with st.form("add_lead_form"):
            new_lead_name = st.text_input("Lead Name*", placeholder="John Doe")
            new_lead_role = st.text_input("Lead Role", placeholder="CTO, Manager, etc.")
            new_lead_channel = st.selectbox("Channel", ["Email", "LinkedIn"])
            new_lead_email = st.text_input("Email*", placeholder="john@example.com")
            new_previous_messages = st.text_area("Previous Messages", placeholder="Optional context...")
            new_response_prob = st.slider("Response Probability", 0.0, 1.0, 0.5, 0.1)
            
            submitted = st.form_submit_button("Add Lead", type="primary")
            
            if submitted:
                if not new_lead_name.strip():
                    st.error("Lead name is required")
                elif not new_lead_email.strip():
                    st.error("Email is required")
                else:
                    success, message = add_lead_to_csv(
                        new_lead_name.strip(),
                        new_lead_role.strip(),
                        new_lead_channel,
                        new_lead_email.strip(),
                        new_previous_messages.strip(),
                        new_response_prob
                    )
                    
                    if success:
                        st.success(message)
                        # Reload leads
                        leads = load_leads()
                        globals()['leads'] = leads
                        env.update_leads(leads)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Current leads display
    if leads:
        st.subheader(f"📊 Current Leads ({len(leads)})")
        for i, lead in enumerate(leads):
            with st.expander(f"{lead.get('lead_name', 'Unknown')} ({lead.get('lead_role', 'No role')})"):
                st.write(f"**Email:** {lead.get('email', 'No email')}")
                st.write(f"**Channel:** {lead.get('channel', 'Unknown')}")
                if lead.get('previous_messages'):
                    st.write(f"**Previous:** {lead.get('previous_messages')}")
    else:
        st.warning("⚠️ No leads available. Please add a lead first.")
    
    st.markdown("---")
    st.header("Agent → Generate & Send")
    
    if not leads:
        st.warning("⚠️ Please add at least one lead before sending messages.")
    else:
        channel = st.selectbox("Channel", ["Email", "LinkedIn"], index=0)
        topic = st.text_input("Topic / Hook", value="how we can reduce time on X")
        lead_names = [l.get("lead_name", "") for l in leads]
        selected_index = st.selectbox("Choose lead", range(len(lead_names)), format_func=lambda i: lead_names[i])
        selected_lead = leads[selected_index]
        lead_email = st.text_input("Lead email (override)", value=selected_lead.get("email", ""))
        lead_name = st.text_input("Lead name (override)", value=selected_lead.get("lead_name", ""))

        # Only ensure in database, not CSV (CSV is managed separately)
        if lead_email:
            ensure_lead_exists_in_db_only(conn, lead_name or "Unknown", lead_email)

        # Combined Generate & Send button for Email
        if channel == "Email":
            subject = st.text_input("Email subject", value=f"Quick note: {topic}")
            
            if st.button("🚀 Generate & Send Email", type="primary"):
                if not lead_email:
                    st.error("Please provide the lead's email.")
                else:
                    # Generate message
                    with st.spinner("Generating message..."):
                        body = generate_agent_message(lead_name, channel, topic)
                        st.session_state["agent_message"] = body
                        st.session_state["agent_channel"] = channel
                        st.session_state["agent_lead"] = lead_name
                        st.session_state["agent_topic"] = topic
                        st.session_state["agent_message_time"] = time.time()
                    
                    # Send email
                    with st.spinner("Sending email..."):
                        ses = STATIC_EMAIL_SETTINGS
                        ok, info = send_email_smtp(
                            ses["smtp_server"],
                            int(ses["smtp_port"]),
                            bool(ses["use_ssl"]),
                            ses["from_email"],
                            ses["password"],
                            lead_email,
                            subject,
                            body,
                        )
                        
                        if ok:
                            st.success("✅ Email generated and sent successfully!")
                            # Log the sent message
                            ts = time.time()
                            classification = "sent"
                            reward = 0.0
                            deal_quality = compute_deal_quality(selected_lead, "ignore")
                            c.execute(
                                "INSERT INTO logs (ts, lead_name, channel, agent_message, human_reply, classification, reward, deal_closed, deal_quality) VALUES (?,?,?,?,?,?,?,?,?)",
                                (ts, lead_name, "Email", body, "", classification, float(reward), 0, float(deal_quality)),
                            )
                            conn.commit()
                            st.session_state["last_sent"] = {
                                "lead_name": lead_name, 
                                "lead_email": lead_email, 
                                "subject": subject, 
                                "body": body, 
                                "ts": ts
                            }
                        else:
                            st.error(f"❌ Failed to send: {info}")
        else:
            # For non-email channels, keep the separate generate button
            if st.button("Generate Message"):
                msg = generate_agent_message(lead_name, channel, topic)
                st.session_state["agent_message"] = msg
                st.session_state["agent_channel"] = channel
                st.session_state["agent_lead"] = lead_name
                st.session_state["agent_topic"] = topic
                st.session_state["agent_message_time"] = time.time()
                st.session_state["pending_reply"] = ""
                st.rerun()

        # Show generated message preview
        if st.session_state.get("agent_message"):
            st.text_area("Generated message preview", value=st.session_state["agent_message"], height=160, disabled=True)

        if st.button("Reset Session State"):
            for k in list(st.session_state.keys()):
                st.session_state.pop(k)
            st.success("Session state cleared.")
            st.rerun()

with col2:
    st.header("📨 Inbox Monitor")
    
    # Manual check inbox button
    if st.button("🔍 Manual Inbox Check"):
        with st.spinner("Checking inbox..."):
            processed = auto_check_inbox()
            if processed == 0:
                st.info("No new messages found.")
    
    # Auto-refresh status
    if st.session_state.get("auto_refresh_enabled"):
        st.success("🔄 Auto-refresh is ENABLED (every 10 seconds)")
    else:
        st.warning("⏸️ Auto-refresh is DISABLED")
    
    st.markdown("---")
    st.markdown("Quick-reply examples (manual testing):")
    if st.button("✅ Book meeting"): st.session_state["pending_reply"] = "Yes, let's schedule a call. Tomorrow works."; st.rerun()
    if st.button("💬 Interested"): st.session_state["pending_reply"] = "Sounds interesting — can you send more details?"; st.rerun()
    if st.button("💸 Objection"): st.session_state["pending_reply"] = "Looks good but budget is tight — is pricing flexible?"; st.rerun()
    if st.button("✖️ Reject"): st.session_state["pending_reply"] = "Not interested, thanks."; st.rerun()
    if st.button("… Ignore"): st.session_state["pending_reply"] = ""; st.rerun()

# ---------------------------
# METRICS
# ---------------------------
st.markdown("---")
logs_df = pd.read_sql_query("SELECT * FROM logs ORDER BY ts DESC", conn)
total_interactions = len(logs_df)
deals_closed = int(logs_df["deal_closed"].sum()) if total_interactions > 0 else 0
avg_deal_quality = float(logs_df["deal_quality"].mean()) if total_interactions > 0 else 0.0
success_rate = (deals_closed / total_interactions * 100) if total_interactions > 0 else 0.0

colA, colB, colC, colD = st.columns(4)
colA.metric("Total Interactions", total_interactions)
colB.metric("Deals Closed", deals_closed)
colC.metric("Success Rate", f"{success_rate:.1f}%")
colD.metric("Avg Deal Quality", f"{avg_deal_quality:.2f}")

if total_interactions > 0:
    breakdown = logs_df["classification"].value_counts().rename_axis("classification").reset_index(name="count")
    st.bar_chart(breakdown.set_index("classification"))
    display_df = logs_df.copy()
    display_df["time"] = pd.to_datetime(display_df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display_df.head(200), use_container_width=True)
else:
    st.info("No interactions yet.")