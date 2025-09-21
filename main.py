"""
Interactive SDR Simulator with real Email send/receive capability.
Enhanced with product-specific messaging and negotiator agent.
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
# PRODUCT CATALOG WITH PRICING
# ---------------------------
PRODUCTS = {
    "MacBook Pro": {
        "hooks": [
            "Good tech shouldn't break the bank — what's fair for you on this MacBook Pro?",
            "What would make this MacBook Pro a no-brainer for you?",
            "How close can we get to your ideal price on a MacBook Pro today?"
        ],
        "base_price": 199000,  # INR
        "min_price": 175000,   # INR - minimum negotiable price
        "description": "Latest MacBook Pro with M3 chip, perfect for professionals"
    },
    "Sony 4K TV": {
        "hooks": [
            "Thinking about upgrading to a Sony 4K TV? Let's make the price work for you.",
            "Why pay full price for a Sony 4K TV when you can negotiate a fair deal?",
            "How we can make that Sony 4K TV fit perfectly into your budget"
        ],
        "base_price": 85000,   # INR
        "min_price": 75000,    # INR
        "description": "55-inch Sony 4K TV with HDR and smart features"
    },
    "iPhone": {
        "hooks": [
            "Your next iPhone upgrade doesn't have to cost a fortune — what's fair for you?",
            "If the price was right, would you pick your iPhone upgrade today?",
            "Let's turn that iPhone upgrade into a deal that makes sense for you"
        ],
        "base_price": 79900,   # INR
        "min_price": 72000,    # INR
        "description": "Latest iPhone with advanced camera and performance"
    },
    "Chair": {
        "hooks": [
            "Comfort shouldn't come with a premium — what would be fair on this chair?",
            "Looking to upgrade your workspace? Let's find a fair deal on this chair.",
            "How we can make a comfortable chair fit your budget without the premium"
        ],
        "base_price": 25000,   # INR
        "min_price": 20000,    # INR
        "description": "Ergonomic office chair with premium comfort and support"
    }
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
            deal_quality REAL,
            product_name TEXT,
            current_price REAL,
            negotiation_stage TEXT
        )"""
    )
    
    # Create leads table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS leads(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            current_product TEXT,
            current_price REAL,
            negotiation_stage TEXT
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
        "ts", "lead_name", "channel", "agent_message", "human_reply",
        "classification", "reward", "deal_closed", "deal_quality",
        "product_name", "current_price", "negotiation_stage"
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
                deal_quality REAL,
                product_name TEXT,
                current_price REAL,
                negotiation_stage TEXT
            )"""
        )
        
        # Create leads table
        c.execute(
            """CREATE TABLE IF NOT EXISTS leads(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                current_product TEXT,
                current_price REAL,
                negotiation_stage TEXT
            )"""
        )
        conn.commit()
    else:
        st.info("📭 No pending replies. New customer replies will appear here for manual negotiation.")
    
    st.markdown("---")
    st.markdown("**Quick-reply examples (for testing):**")
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("✅ Accept Deal"): 
            st.session_state["pending_reply"] = "Yes, I'll take it at that price. Let's proceed!"
            st.rerun()
        if st.button("💬 Show Interest"): 
            st.session_state["pending_reply"] = "This looks interesting. Can you tell me more about the features?"
            st.rerun()
        if st.button("📅 Book Meeting"): 
            st.session_state["pending_reply"] = "Let's schedule a call to discuss this further."
            st.rerun()
    
    with col_b:
        if st.button("💸 Price Objection"): 
            st.session_state["pending_reply"] = "The price seems a bit high. Can you do better? My budget is around ₹50,000."
            st.rerun()
        if st.button("❌ Reject"): 
            st.session_state["pending_reply"] = "Not interested, thanks."
            st.rerun()
        if st.button("🤔 Neutral"): 
            st.session_state["pending_reply"] = "I need to think about this."
            st.rerun()
        existing_col_names = [col[1] for col in cols]
        if len(existing_col_names) < len(expected_col_names):
            # Add missing columns
            migration_performed = True
            missing_cols = [col for col in expected_col_names if col not in existing_col_names]
            for col in missing_cols:
                if col in ["product_name", "negotiation_stage"]:
                    c.execute(f"ALTER TABLE logs ADD COLUMN {col} TEXT")
                elif col == "current_price":
                    c.execute(f"ALTER TABLE logs ADD COLUMN {col} REAL")
            
            # Also update leads table
            c.execute("PRAGMA table_info(leads)")
            leads_cols = [col[1] for col in c.fetchall()]
            if "current_product" not in leads_cols:
                c.execute("ALTER TABLE leads ADD COLUMN current_product TEXT")
            if "current_price" not in leads_cols:
                c.execute("ALTER TABLE leads ADD COLUMN current_price REAL")
            if "negotiation_stage" not in leads_cols:
                c.execute("ALTER TABLE leads ADD COLUMN negotiation_stage TEXT")
            
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
# DB utility: ensure lead exists with product tracking
# ---------------------------
def ensure_lead_exists_in_db_only(conn, lead_name, lead_email, product_name=None, current_price=None):
    """Ensure lead exists in database with product tracking"""
    if not lead_email:
        return
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM leads WHERE email = ?", (lead_email,))
        row = cursor.fetchone()
        exists = row[0] if row else 0
        if not exists:
            cursor.execute(
                "INSERT OR IGNORE INTO leads (name, email, current_product, current_price, negotiation_stage) VALUES (?, ?, ?, ?, ?)",
                (lead_name, lead_email, product_name, current_price, "initial")
            )
        else:
            # Update product info if provided
            if product_name:
                cursor.execute(
                    "UPDATE leads SET current_product = ?, current_price = ?, negotiation_stage = ? WHERE email = ?",
                    (product_name, current_price, "negotiating", lead_email)
                )
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error in ensure_lead_exists: {e}")
    except Exception as e:
        st.warning(f"Error ensuring lead exists: {e}")

def get_lead_negotiation_state(conn, lead_email):
    """Get current negotiation state for a lead"""
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT current_product, current_price, negotiation_stage FROM leads WHERE email = ?", 
            (lead_email,)
        )
        result = cursor.fetchone()
        if result:
            return {
                "product": result[0],
                "price": result[1],
                "stage": result[2]
            }
        return None
    except Exception:
        return None

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
# PRODUCT-AWARE MESSAGE GENERATOR
# ---------------------------
def generate_product_intro_message(lead_name: str, channel: str) -> Tuple[str, str]:
    """Generate initial product introduction message"""
    # Randomly select a product for initial outreach
    selected_product = random.choice(list(PRODUCTS.keys()))
    product_info = PRODUCTS[selected_product]
    
    # Select random hook for the product
    hook = random.choice(product_info["hooks"])
    
    if COHERE_INSTALLED and COHERE_API_KEY:
        try:
            prompt = (
                f"{PERSONA_INSTRUCTION}\n\n"
                f"Write a short, friendly, and professional {channel} message to {lead_name}. "
                f"Use this hook: '{hook}' "
                f"Present the {selected_product} ({product_info['description']}) "
                f"at ₹{product_info['base_price']:,} but emphasize that pricing is flexible and we can work together. "
                f"Make it warm and conversational. End with my name, Sasitharan."
            )
            client = cohere.Client(COHERE_API_KEY)
            resp = client.chat(model=PRIMARY_MODEL, message=prompt, temperature=0.7)
            if resp.text.strip():
                return resp.text.strip(), selected_product
        except Exception:
            pass
    
    # Fallback message
    message = (
        f"Hi {lead_name},\n\n"
        f"{hook}\n\n"
        f"I have a {selected_product} ({product_info['description']}) "
        f"at ₹{product_info['base_price']:,}, but I'm flexible on pricing. "
        f"Would you be interested in discussing this?\n\n"
        f"Best regards,\n"
        f"Sasitharan"
    )
    
    return message, selected_product

# ---------------------------
# ENHANCED NEGOTIATOR MESSAGE GENERATOR WITH PROFIT MARGINS
# ---------------------------
def generate_negotiator_message(lead_name: str, reply_text: str, product_name: str, current_price: float, lead_email: str) -> Tuple[str, float]:
    """Generate negotiator response with price adjustment and profit margin protection"""
    product_info = PRODUCTS.get(product_name, {})
    base_price = product_info.get("base_price", current_price)
    min_price = product_info.get("min_price", current_price * 0.8)
    
    # Calculate profit margins
    cost_price = min_price * 0.7  # Assuming cost is 70% of minimum price
    current_profit_margin = ((current_price - cost_price) / current_price) * 100
    min_profit_margin = ((min_price - cost_price) / min_price) * 100
    
    # Analyze the reply to determine price strategy
    reply_lower = reply_text.lower()
    
    # Extract any price mentioned in the reply
    import re
    price_matches = re.findall(r'₹?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', reply_text)
    customer_price = None
    if price_matches:
        try:
            # Take the largest number as potential price
            customer_price = max([float(p.replace(',', '')) for p in price_matches])
        except:
            customer_price = None
    
    # Determine new offer price with profit margin protection
    if customer_price and customer_price >= min_price:
        # Customer mentioned a reasonable price
        new_price = max(customer_price, min_price)
        price_strategy = "accept_customer_price"
    elif "expensive" in reply_lower or "budget" in reply_lower or "price" in reply_lower:
        # Customer objects to price - offer discount but protect margins
        if current_price > min_price * 1.2:  # Only discount if we have room
            discount = random.uniform(0.08, 0.15)  # 8-15% discount
            new_price = max(current_price * (1 - discount), min_price)
        else:
            discount = random.uniform(0.02, 0.05)  # Small discount near minimum
            new_price = max(current_price * (1 - discount), min_price)
        price_strategy = "discount_offer"
    elif any(word in reply_lower for word in ["interested", "good", "like", "tell me more"]):
        # Customer shows interest - hold price or small discount
        discount = random.uniform(0.02, 0.08)  # 2-8% discount
        new_price = max(current_price * (1 - discount), min_price)
        price_strategy = "hold_price"
    else:
        # Neutral response - moderate discount
        discount = random.uniform(0.05, 0.10)  # 5-10% discount
        new_price = max(current_price * (1 - discount), min_price)
        price_strategy = "moderate_discount"
    
    new_price = round(new_price, -2)  # Round to nearest 100
    new_profit_margin = ((new_price - cost_price) / new_price) * 100
    
    if COHERE_INSTALLED and COHERE_API_KEY:
        try:
            # Add profit margin context to the prompt
            margin_info = f"Maintain at least {min_profit_margin:.1f}% profit margin. Current offer maintains {new_profit_margin:.1f}% margin."
            
            prompt = (
                f"You are a skilled sales negotiator for {product_name}. "
                f"Customer {lead_name} replied: \"{reply_text}\" "
                f"Current offer: ₹{current_price:,.0f}, New offer: ₹{new_price:,.0f} "
                f"Strategy: {price_strategy}. {margin_info} "
                f"Write a persuasive, warm response (2-3 sentences) that: "
                f"1) Acknowledges their concern positively "
                f"2) Presents the new price of ₹{new_price:,.0f} as a special offer "
                f"3) Highlights value and creates urgency "
                f"4) Asks for decision or next step "
                f"Keep it conversational and professional. End with: Best regards, Sasitharan"
            )
            
            client = cohere.Client(COHERE_API_KEY)
            resp = client.chat(model=PRIMARY_MODEL, message=prompt, temperature=0.7)
            if resp.text.strip():
                # Update lead negotiation state
                ensure_lead_exists_in_db_only(conn, lead_name, lead_email, product_name, new_price)
                return resp.text.strip(), new_price
        except Exception as e:
            st.warning(f"Cohere error: {e}")
    
    # Fallback negotiator response with profit margin awareness
    margin_comment = "This is my best price maintaining quality" if new_price <= min_price * 1.1 else "I can offer you this special price"
    
    responses_by_strategy = {
        "accept_customer_price": [
            f"Hi {lead_name}, that works perfectly! ₹{new_price:,.0f} is a fair price for this {product_name}. {margin_comment}. Shall we move forward?",
            f"Excellent point, {lead_name}! I can do ₹{new_price:,.0f} for you. {margin_comment}. Ready to finalize this?"
        ],
        "discount_offer": [
            f"I understand your budget concern, {lead_name}. Let me offer you ₹{new_price:,.0f} - {margin_comment}. How does that work for you?",
            f"Great point about the budget, {lead_name}. I can bring it down to ₹{new_price:,.0f}. {margin_comment}. Is this more workable?"
        ],
        "hold_price": [
            f"I'm glad you're interested, {lead_name}! At ₹{new_price:,.0f}, this {product_name} offers excellent value. {margin_comment}. Shall we proceed?",
            f"Perfect, {lead_name}! ₹{new_price:,.0f} for this {product_name} is an excellent investment. {margin_comment}. Ready to move forward?"
        ],
        "moderate_discount": [
            f"Thanks for your interest, {lead_name}! I can offer you ₹{new_price:,.0f} - {margin_comment}. What do you think?",
            f"Hi {lead_name}, I can work with ₹{new_price:,.0f} for you. {margin_comment} for this quality. Interested?"
        ]
    }
    
    response_options = responses_by_strategy.get(price_strategy, responses_by_strategy["moderate_discount"])
    message = random.choice(response_options) + f"\n\nBest regards,\nSasitharan"
    
    # Update lead negotiation state
    ensure_lead_exists_in_db_only(conn, lead_name, lead_email, product_name, new_price)
    
    return message, new_price

# ---------------------------
# ENHANCED REPLY CLASSIFIER
# ---------------------------
def classify_reply(text: str) -> str:
    if not text or text.strip() == "":
        return "ignore"
    
    t = text.lower()
    
    # Check for deal closure
    if any(kw in t for kw in ["buy", "purchase", "take it", "deal", "sold", "yes let's do", "i'll take", "agreed"]):
        return "deal_closed"
    
    # Check for booking/meeting
    if any(kw in t for kw in ["book", "schedule", "call", "meeting", "yes", "ok", "tomorrow", "next week"]):
        return "book"
    
    # Check for price negotiation/objection
    if any(kw in t for kw in ["price", "cost", "budget", "expensive", "cheaper", "discount", "₹", "rupees"]):
        return "price_negotiation"
    
    # Check for product interest
    if any(kw in t for kw in ["interested", "tell me more", "more info", "details", "specs", "features"]):
        return "interested"
    
    # Check for rejection
    if any(kw in t for kw in ["no", "not interested", "don't want", "no thanks", "uninterested", "pass"]):
        return "reject"
    
    return "neutral"

# ---------------------------
# DEAL QUALITY (enhanced)
# ---------------------------
def compute_deal_quality(lead_record: Dict, classification: str, product_name: str = None) -> float:
    role = str(lead_record.get("lead_role", "")).lower()
    role_weight = 0.5 if any(k in role for k in ["cto", "founder", "ceo", "manager", "director"]) else 0.3
    
    prior_msg = str(lead_record.get("previous_messages", "") or "")
    prior = 0.2 if prior_msg.strip() else 0.0
    
    # Enhanced classification bonuses
    class_bonus = {
        "deal_closed": 1.0,
        "book": 0.8,
        "price_negotiation": 0.6,
        "interested": 0.4,
        "neutral": 0.1,
        "ignore": 0.0,
        "reject": -0.3
    }.get(classification, 0.0)
    
    # Product value bonus (higher value products get slight boost)
    product_bonus = 0.0
    if product_name and product_name in PRODUCTS:
        base_price = PRODUCTS[product_name]["base_price"]
        if base_price > 100000:  # High-value products
            product_bonus = 0.1
    
    return max(0.0, min(1.0, role_weight + prior + class_bonus + product_bonus))

# ---------------------------
# EMAIL HELPERS (SMTP & IMAP) - unchanged
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
# Map reply -> lead (unchanged)
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
# ENHANCED Auto-refresh inbox function
# ---------------------------
def auto_check_inbox():
    """Function to automatically check inbox and process replies with negotiation logic"""
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
        st.success(f"📧 Auto-refresh: Found {len(replies)} new message(s)")
        processed = 0
        
        for rep in replies:
            lead = map_reply_to_lead(rep, leads)
            reply_body = rep.get("body", "")
            
            # Handle unknown senders
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
                lead_email = lead.get("email")
                lead_name = lead.get("lead_name")
                
                # Get current negotiation state
                negotiation_state = get_lead_negotiation_state(conn, lead_email)
                
                # Classify the reply
                classification = classify_reply(reply_body)
                
                # Store reply for manual processing - NO AUTO-RESPONSE
                current_product = negotiation_state["product"] if negotiation_state else None
                current_price = negotiation_state["price"] if negotiation_state else None
                
                if classification == "reject":
                    # Customer rejected - close the deal
                    reward = -2.0
                    deal_closed = -1  # Mark as rejected
                    negotiation_stage = "rejected"
                    
                elif classification == "deal_closed":
                    # Customer agreed to buy
                    reward = 5.0
                    deal_closed = 1
                    negotiation_stage = "closed_won"
                    
                elif classification in ["price_negotiation", "interested"] and current_product:
                    # Continue negotiation (but don't auto-respond)
                    reward = 1.0
                    deal_closed = 0
                    negotiation_stage = "negotiating"
                        
                else:
                    # Neutral or other responses
                    reward = 0.0
                    deal_closed = 0
                    negotiation_stage = negotiation_state["stage"] if negotiation_state else "initial"
                
                # Compute deal quality
                deal_quality = compute_deal_quality(lead, classification, current_product)
                
                # Get the last agent message for logging
                agent_message = ""
                try:
                    q = c.execute(
                        "SELECT agent_message FROM logs WHERE lead_name=? AND channel='Email' AND agent_message<>'' ORDER BY ts DESC LIMIT 1",
                        (lead_name,),
                    )
                    row = q.fetchone()
                    if row:
                        agent_message = row[0]
                except Exception:
                    agent_message = ""
                
                # Log the interaction
                ts = time.time()
                c.execute(
                    """INSERT INTO logs (ts, lead_name, channel, agent_message, human_reply, classification, 
                       reward, deal_closed, deal_quality, product_name, current_price, negotiation_stage) 
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ts, lead_name, "Email", agent_message, reply_body, classification, 
                     float(reward), int(deal_closed), float(deal_quality), current_product, current_price, negotiation_stage),
                )
                conn.commit()
                
                processed += 1
                
                # Show the processed reply
                status_emoji = {
                    "deal_closed": "🎉",
                    "reject": "❌", 
                    "price_negotiation": "💰",
                    "interested": "👍",
                    "book": "📅",
                    "neutral": "💬"
                }.get(classification, "📧")
                
                with st.expander(f"{status_emoji} New Reply from {lead_name} - {classification.upper().replace('_', ' ')}", expanded=True):
                    st.markdown(f"**From:** {rep.get('from')}  \n**Subject:** {rep.get('subject')}")
                    st.write("**Reply:**")
                    st.write(reply_body[:1000])
                    if current_product:
                        st.info(f"**Product:** {current_product} | **Current Price:** ₹{current_price:,.0f}" if current_price else f"**Product:** {current_product}")
                    
                    # Store this reply for manual response generation
                    st.session_state[f"pending_reply_{lead_email}"] = {
                        "lead_name": lead_name,
                        "lead_email": lead_email,
                        "reply_body": reply_body,
                        "classification": classification,
                        "product": current_product,
                        "price": current_price,
                        "subject": rep.get('subject', '')
                    }
        
        return processed
    return 0

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Enhanced SDR Simulator — Product Sales & Negotiation", layout="wide")
st.title("🤖 Enhanced SDR Simulator — Product Sales with AI Negotiator")

# Initialize session state
if "pending_reply" not in st.session_state:
    st.session_state["pending_reply"] = ""

if "last_inbox_check" not in st.session_state:
    st.session_state["last_inbox_check"] = 0

if "auto_refresh_enabled" not in st.session_state:
    st.session_state["auto_refresh_enabled"] = True

if "selected_product" not in st.session_state:
    st.session_state["selected_product"] = list(PRODUCTS.keys())[0]

# Auto-refresh inbox every 10 seconds
current_time = time.time()
if (current_time - st.session_state["last_inbox_check"]) >= 10 and st.session_state.get("auto_refresh_enabled", True):
    st.session_state["last_inbox_check"] = current_time
    auto_check_inbox()
    st.rerun()

# Sidebar - Show static email settings and product info
with st.sidebar:
    st.header("🛍️ Product Catalog")
    
    # Display products with prices
    for product, info in PRODUCTS.items():
        with st.expander(f"{product} - ₹{info['base_price']:,}"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Base Price:** ₹{info['base_price']:,}")
            st.write(f"**Min Price:** ₹{info['min_price']:,}")
            st.write("**Sample Hooks:**")
            for hook in info['hooks'][:2]:  # Show first 2 hooks
                st.write(f"• {hook}")
    
    st.markdown("---")
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
            lead_email = lead.get('email', '')
            negotiation_state = get_lead_negotiation_state(conn, lead_email) if lead_email else None
            
            with st.expander(f"{lead.get('lead_name', 'Unknown')} ({lead.get('lead_role', 'No role')})"):
                st.write(f"**Email:** {lead_email}")
                st.write(f"**Channel:** {lead.get('channel', 'Unknown')}")
                if negotiation_state and negotiation_state['product']:
                    st.write(f"**Current Product:** {negotiation_state['product']}")
                    if negotiation_state['price']:
                        st.write(f"**Current Price:** ₹{negotiation_state['price']:,.0f}")
                    st.write(f"**Stage:** {negotiation_state['stage']}")
                if lead.get('previous_messages'):
                    st.write(f"**Previous:** {lead.get('previous_messages')}")
    else:
        st.warning("⚠️ No leads available. Please add a lead first.")
    
    st.markdown("---")
    st.header("🚀 Product Outreach")
    
    if not leads:
        st.warning("⚠️ Please add at least one lead before sending messages.")
    else:
        # Lead selection
        lead_names = [l.get("lead_name", "") for l in leads]
        selected_index = st.selectbox("Choose lead", range(len(lead_names)), format_func=lambda i: lead_names[i])
        selected_lead = leads[selected_index]
        lead_email = st.text_input("Lead email (override)", value=selected_lead.get("email", ""))
        lead_name = st.text_input("Lead name (override)", value=selected_lead.get("lead_name", ""))

        # Product selection option
        product_mode = st.radio("Product Selection", ["Auto (Random)", "Manual"])
        
        if product_mode == "Manual":
            st.session_state["selected_product"] = st.selectbox(
                "Select Product", 
                list(PRODUCTS.keys()),
                index=list(PRODUCTS.keys()).index(st.session_state.get("selected_product", list(PRODUCTS.keys())[0]))
            )

        # Email subject
        subject = st.text_input("Email subject", value="Special offer just for you")
        
        # Generate & Send Product Introduction
        if st.button("🎯 Generate & Send Product Introduction", type="primary"):
            if not lead_email:
                st.error("Please provide the lead's email.")
            else:
                with st.spinner("Generating product introduction..."):
                    if product_mode == "Auto (Random)":
                        body, selected_product = generate_product_intro_message(lead_name, "Email")
                    else:
                        # Manual product selection
                        selected_product = st.session_state["selected_product"]
                        product_info = PRODUCTS[selected_product]
                        hook = random.choice(product_info["hooks"])
                        
                        if COHERE_INSTALLED and COHERE_API_KEY:
                            try:
                                prompt = (
                                    f"{PERSONA_INSTRUCTION}\n\n"
                                    f"Write a short, friendly, and professional email to {lead_name}. "
                                    f"Use this hook: '{hook}' "
                                    f"Present the {selected_product} ({product_info['description']}) "
                                    f"at ₹{product_info['base_price']:,} but emphasize that pricing is flexible. "
                                    f"Make it warm and conversational. End with my name, Sasitharan."
                                )
                                client = cohere.Client(COHERE_API_KEY)
                                resp = client.chat(model=PRIMARY_MODEL, message=prompt, temperature=0.7)
                                body = resp.text.strip() if resp.text.strip() else f"Hi {lead_name},\n\n{hook}\n\nBest regards,\nSasitharan"
                            except Exception:
                                body = f"Hi {lead_name},\n\n{hook}\n\nBest regards,\nSasitharan"
                        else:
                            body = f"Hi {lead_name},\n\n{hook}\n\nBest regards,\nSasitharan"
                
                # Send email
                with st.spinner("Sending product introduction..."):
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
                        st.success(f"✅ Product introduction sent for {selected_product}!")
                        
                        # Update lead in database with product info
                        ensure_lead_exists_in_db_only(conn, lead_name, lead_email, selected_product, PRODUCTS[selected_product]["base_price"])
                        
                        # Log the sent message
                        ts = time.time()
                        c.execute(
                            """INSERT INTO logs (ts, lead_name, channel, agent_message, human_reply, classification, 
                               reward, deal_closed, deal_quality, product_name, current_price, negotiation_stage) 
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (ts, lead_name, "Email", body, "", "product_intro", 0.0, 0, 0.0, 
                             selected_product, PRODUCTS[selected_product]["base_price"], "initial"),
                        )
                        conn.commit()
                        
                        st.session_state["last_sent"] = {
                            "lead_name": lead_name, 
                            "lead_email": lead_email, 
                            "subject": subject, 
                            "body": body, 
                            "product": selected_product,
                            "ts": ts
                        }
                    else:
                        st.error(f"❌ Failed to send: {info}")

        # Show last sent message
        if st.session_state.get("last_sent"):
            st.text_area("Last sent message preview", value=st.session_state["last_sent"]["body"], height=160, disabled=True)

        if st.button("Reset Session State"):
            for k in list(st.session_state.keys()):
                st.session_state.pop(k)
            st.success("Session state cleared.")
            st.rerun()

with col2:
    st.header("📨 Manual Negotiation Center")
    
    # Manual check inbox button
    if st.button("🔍 Manual Inbox Check"):
        with st.spinner("Checking inbox..."):
            processed = auto_check_inbox()
            if processed == 0:
                st.info("No new messages found.")
    
    # Auto-refresh status
    if st.session_state.get("auto_refresh_enabled"):
        st.success("🔄 Auto-refresh is ENABLED (every 10 seconds)")
        st.info("📧 New replies will be detected automatically - respond manually below")
    else:
        st.warning("⏸️ Auto-refresh is DISABLED")
    
    st.markdown("---")
    
    # Manual Negotiator Response Section
    st.subheader("🤖 AI Negotiator Responses")
    
    # Check for pending replies that need responses
    pending_replies = []
    for key in st.session_state.keys():
        if key.startswith("pending_reply_") and st.session_state[key]:
            pending_replies.append(st.session_state[key])
    
    if pending_replies:
        st.success(f"📬 {len(pending_replies)} reply(s) ready for negotiation!")
        
        for reply_data in pending_replies:
            lead_name = reply_data["lead_name"]
            lead_email = reply_data["lead_email"]
            classification = reply_data["classification"]
            product = reply_data["product"]
            current_price = reply_data["price"]
            reply_body = reply_data["reply_body"]
            
            with st.expander(f"🎯 Negotiate with {lead_name} - {classification.upper().replace('_', ' ')}", expanded=True):
                st.write(f"**Product:** {product}")
                if current_price:
                    # Calculate profit margin info
                    if product and product in PRODUCTS:
                        min_price = PRODUCTS[product]["min_price"]
                        cost_price = min_price * 0.7
                        profit_margin = ((current_price - cost_price) / current_price) * 100
                        st.info(f"**Current Price:** ₹{current_price:,.0f} | **Profit Margin:** {profit_margin:.1f}%")
                    else:
                        st.info(f"**Current Price:** ₹{current_price:,.0f}")
                
                st.write("**Customer Reply:**")
                st.write(f'"{reply_body[:300]}..."' if len(reply_body) > 300 else f'"{reply_body}"')
                
                # Generate negotiation response
                col_gen, col_send = st.columns(2)
                
                with col_gen:
                    if st.button(f"🧠 Generate AI Response", key=f"gen_{lead_email}"):
                        if product and current_price:
                            with st.spinner("Generating negotiation response..."):
                                try:
                                    response_msg, new_price = generate_negotiator_message(
                                        lead_name, reply_body, product, current_price, lead_email
                                    )
                                    
                                    # Store the generated response
                                    st.session_state[f"generated_response_{lead_email}"] = {
                                        "message": response_msg,
                                        "new_price": new_price,
                                        "subject": f"Re: {reply_data.get('subject', 'Our conversation')}"
                                    }
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating response: {e}")
                        else:
                            st.error("Missing product or price information")
                
                # Show generated response if available
                if f"generated_response_{lead_email}" in st.session_state:
                    response_data = st.session_state[f"generated_response_{lead_email}"]
                    
                    st.write("**Generated AI Response:**")
                    st.text_area("AI Response:", value=response_data["message"], height=150, key=f"preview_{lead_email}")
                    
                    # Show price change
                    if current_price and response_data["new_price"] != current_price:
                        price_change = response_data["new_price"] - current_price
                        price_color = "red" if price_change < 0 else "green"
                        st.markdown(f"**Price Change:** <span style='color:{price_color}'>₹{price_change:+,.0f}</span> (New: ₹{response_data['new_price']:,.0f})", unsafe_allow_html=True)
                    
                    with col_send:
                        if st.button(f"📤 Send Response", key=f"send_{lead_email}", type="primary"):
                            with st.spinner("Sending negotiation response..."):
                                ses = STATIC_EMAIL_SETTINGS
                                success, error_msg = send_email_smtp(
                                    ses["smtp_server"],
                                    int(ses["smtp_port"]),
                                    bool(ses["use_ssl"]),
                                    ses["from_email"],
                                    ses["password"],
                                    lead_email,
                                    response_data["subject"],
                                    response_data["message"],
                                )
                                
                                if success:
                                    st.success("✅ Negotiation response sent!")
                                    
                                    # Log the sent response
                                    ts = time.time()
                                    c.execute(
                                        """INSERT INTO logs (ts, lead_name, channel, agent_message, human_reply, classification, 
                                           reward, deal_closed, deal_quality, product_name, current_price, negotiation_stage) 
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                        (ts, lead_name, "Email", response_data["message"], "", "negotiation_response", 
                                         0.0, 0, 0.0, product, response_data["new_price"], "negotiating"),
                                    )
                                    conn.commit()
                                    
                                    # Clear the pending reply and generated response
                                    del st.session_state[f"pending_reply_{lead_email}"]
                                    del st.session_state[f"generated_response_{lead_email}"]
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to send: {error_msg}")
                
                # Clear this reply button
                if st.button(f"🗑️ Clear Reply", key=f"clear_{lead_email}"):
                    del st.session_state[f"pending_reply_{lead_email}"]
                    if f"generated_response_{lead_email}" in st.session_state:
                        del st.session_state[f"generated_response_{lead_email}"]
                    st.rerun()
    else:
        st.info("No pending replies for negotiation.")

# ---------------------------
# ENHANCED METRICS
# ---------------------------
st.markdown("---")
st.header("📊 Enhanced Sales Metrics")

logs_df = pd.read_sql_query("SELECT * FROM logs ORDER BY ts DESC", conn)
total_interactions = len(logs_df)
deals_closed = int(logs_df[logs_df["deal_closed"] == 1]["deal_closed"].sum()) if total_interactions > 0 else 0
deals_rejected = int(logs_df[logs_df["deal_closed"] == -1]["deal_closed"].count()) if total_interactions > 0 else 0
avg_deal_quality = float(logs_df["deal_quality"].mean()) if total_interactions > 0 else 0.0
success_rate = (deals_closed / total_interactions * 100) if total_interactions > 0 else 0.0

# Calculate total revenue and profit
total_revenue = 0
total_profit = 0
if total_interactions > 0:
    closed_deals = logs_df[logs_df["deal_closed"] == 1]
    if not closed_deals.empty and "current_price" in closed_deals.columns:
        total_revenue = closed_deals["current_price"].sum()
        # Calculate profit (assuming cost is 70% of minimum price)
        for _, deal in closed_deals.iterrows():
            if deal["product_name"] in PRODUCTS:
                min_price = PRODUCTS[deal["product_name"]]["min_price"]
                cost_price = min_price * 0.7
                profit = deal["current_price"] - cost_price
                total_profit += profit

colA, colB, colC, colD, colE, colF = st.columns(6)
colA.metric("Total Interactions", total_interactions)
colB.metric("Deals Closed", deals_closed)
colC.metric("Deals Rejected", deals_rejected)
colD.metric("Success Rate", f"{success_rate:.1f}%")
colE.metric("Total Revenue", f"₹{total_revenue:,.0f}")
colF.metric("Total Profit", f"₹{total_profit:,.0f}")

# Product-wise breakdown with profit margins
if total_interactions > 0:
    st.subheader("📈 Analytics Dashboard")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("**Classification Breakdown:**")
        classification_breakdown = logs_df["classification"].value_counts().rename_axis("classification").reset_index(name="count")
        st.bar_chart(classification_breakdown.set_index("classification"))
    
    with col_right:
        st.write("**Product Performance:**")
        product_logs = logs_df[logs_df["product_name"].notna()]
        if not product_logs.empty:
            product_breakdown = product_logs["product_name"].value_counts().rename_axis("product").reset_index(name="interactions")
            st.bar_chart(product_breakdown.set_index("product"))
    
    # Profit Margin Analysis
    st.subheader("💰 Profit Margin Analysis")
    if not closed_deals.empty:
        profit_data = []
        for _, deal in closed_deals.iterrows():
            if deal["product_name"] in PRODUCTS:
                min_price = PRODUCTS[deal["product_name"]]["min_price"]
                cost_price = min_price * 0.7
                profit = deal["current_price"] - cost_price
                profit_margin = (profit / deal["current_price"]) * 100
                profit_data.append({
                    "Product": deal["product_name"],
                    "Sale Price": deal["current_price"],
                    "Profit": profit,
                    "Margin %": profit_margin,
                    "Lead": deal["lead_name"]
                })
        
        if profit_data:
            profit_df = pd.DataFrame(profit_data)
            st.dataframe(profit_df, use_container_width=True)
            
            # Average profit margin by product
            avg_margins = profit_df.groupby("Product")["Margin %"].mean().reset_index()
            st.write("**Average Profit Margins by Product:**")
            st.bar_chart(avg_margins.set_index("Product"))
    else:
        st.info("No closed deals yet to analyze profit margins.")
    
    # Recent interactions table
    st.subheader("📋 Recent Interactions")
    display_df = logs_df.copy()
    display_df["time"] = pd.to_datetime(display_df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Select relevant columns for display
    display_columns = ["time", "lead_name", "classification", "product_name", "current_price", "negotiation_stage", "deal_closed"]
    available_columns = [col for col in display_columns if col in display_df.columns]
    
    st.dataframe(display_df[available_columns].head(50), use_container_width=True)
else:
    st.info("No interactions yet. Send some product introductions to get started!")