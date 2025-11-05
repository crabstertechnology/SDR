"""
Interactive SDR Simulator with real Email send/receive capability.
Enhanced with product-specific messaging and AI-powered negotiator agent.
Modified with static email settings and intelligent deal closure logic.
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

import os
import streamlit as st
# Safe secrets load
try:
    COHERE_API_KEY = st.secrets.get("COHERE_API_KEY")
except Exception:
    COHERE_API_KEY = None

# Fallback to .env or system
if not COHERE_API_KEY:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

if not COHERE_API_KEY:
    st.warning("⚠️ No Cohere API key found. Check .env or Streamlit secrets.")
else:
    import cohere
    co = cohere.Client(COHERE_API_KEY)
    st.success("✅ Cohere AI connected successfully!")


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
# AI-POWERED INTELLIGENT SENTIMENT ANALYZER
# ---------------------------
def analyze_customer_sentiment(reply_text: str) -> Dict[str, any]:
    """Use AI to analyze customer sentiment and intent"""
    
    if COHERE_INSTALLED and COHERE_API_KEY:
        try:
            prompt = (
                f"Analyze this customer reply for sales negotiation context:\n\n"
                f"Customer Reply: \"{reply_text}\"\n\n"
                f"Please analyze and respond with ONLY a JSON object containing:\n"
                f"1. sentiment: 'positive', 'negative', or 'neutral'\n"
                f"2. intent: 'buy_ready', 'price_concern', 'need_info', 'reject', 'schedule_call', 'neutral'\n"
                f"3. urgency: 'high', 'medium', 'low'\n"
                f"4. price_sensitivity: 'high', 'medium', 'low'\n"
                f"5. confidence: float between 0.0-1.0\n\n"
                f"Example: {{\"sentiment\":\"positive\",\"intent\":\"buy_ready\",\"urgency\":\"high\",\"price_sensitivity\":\"low\",\"confidence\":0.85}}\n\n"
                f"Return ONLY the JSON object, no other text."
            )
            
            client = cohere.Client(COHERE_API_KEY)
            resp = client.chat(model=PRIMARY_MODEL, message=prompt, temperature=0.3)
            
            # Try to parse JSON response
            import json
            try:
                analysis = json.loads(resp.text.strip())
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                pass
                
        except Exception as e:
            st.warning(f"AI sentiment analysis failed: {e}")
    
    # Fallback rule-based analysis
    reply_lower = reply_text.lower()
    
    # Determine sentiment
    positive_words = ["yes", "interested", "great", "good", "excellent", "perfect", "agree", "ok", "sure", "definitely"]
    negative_words = ["no", "not interested", "expensive", "can't", "don't", "reject", "pass"]
    
    sentiment = "neutral"
    if any(word in reply_lower for word in positive_words):
        sentiment = "positive"
    elif any(word in reply_lower for word in negative_words):
        sentiment = "negative"
    
    # Determine intent
    intent = "neutral"
    if any(word in reply_lower for word in ["buy", "purchase", "take it", "deal", "sold", "yes let's do"]):
        intent = "buy_ready"
    elif any(word in reply_lower for word in ["price", "cost", "budget", "expensive", "cheaper", "discount"]):
        intent = "price_concern"
    elif any(word in reply_lower for word in ["more info", "tell me more", "details", "specs", "features"]):
        intent = "need_info"
    elif any(word in reply_lower for word in ["not interested", "no thanks", "reject", "pass"]):
        intent = "reject"
    elif any(word in reply_lower for word in ["call", "meeting", "schedule", "discuss", "talk"]):
        intent = "schedule_call"
    
    return {
        "sentiment": sentiment,
        "intent": intent,
        "urgency": "medium",
        "price_sensitivity": "medium",
        "confidence": 0.7
    }

# ---------------------------
# ENHANCED AI-POWERED NEGOTIATOR WITH INTELLIGENT RESPONSES
# ---------------------------
def generate_intelligent_negotiator_message(lead_name: str, reply_text: str, product_name: str, current_price: float, lead_email: str) -> Tuple[str, float, str]:
    """Generate intelligent negotiator response based on AI sentiment analysis"""
    
    # Analyze customer sentiment and intent
    analysis = analyze_customer_sentiment(reply_text)
    sentiment = analysis.get("sentiment", "neutral")
    intent = analysis.get("intent", "neutral")
    urgency = analysis.get("urgency", "medium")
    price_sensitivity = analysis.get("price_sensitivity", "medium")
    
    product_info = PRODUCTS.get(product_name, {})
    base_price = product_info.get("base_price", current_price)
    min_price = product_info.get("min_price", current_price * 0.8)
    
    # Calculate profit margins
    cost_price = min_price * 0.7  # Assuming cost is 70% of minimum price
    
    # Determine response strategy and new price based on AI analysis
    new_price = current_price
    deal_status = "ongoing"  # ongoing, closed_won, closed_lost
    
    if intent == "buy_ready" and sentiment == "positive":
        # Customer is ready to buy - close the deal
        deal_status = "closed_won"
        new_price = current_price  # Keep current offer
        
    elif intent == "reject" and sentiment == "negative":
        # Customer rejected - end gracefully
        deal_status = "closed_lost"
        new_price = current_price
        
    elif intent == "price_concern":
        # Customer has price concerns - negotiate
        if price_sensitivity == "high":
            discount = random.uniform(0.10, 0.20)  # 10-20% discount
        else:
            discount = random.uniform(0.05, 0.12)  # 5-12% discount
        new_price = max(current_price * (1 - discount), min_price)
        
    elif intent == "need_info":
        # Customer needs more information
        discount = random.uniform(0.02, 0.08)  # Small discount to sweeten
        new_price = max(current_price * (1 - discount), min_price)
        
    elif intent == "schedule_call":
        # Customer wants to schedule a call
        discount = random.uniform(0.03, 0.10)  # Moderate discount
        new_price = max(current_price * (1 - discount), min_price)
        
    else:
        # Neutral response
        discount = random.uniform(0.05, 0.10)
        new_price = max(current_price * (1 - discount), min_price)
    
    new_price = round(new_price, -2)  # Round to nearest 100
    new_profit_margin = ((new_price - cost_price) / new_price) * 100
    
    if COHERE_INSTALLED and COHERE_API_KEY:
        try:
            # Create context-aware prompt based on AI analysis
            context = (
                f"Customer sentiment: {sentiment}, Intent: {intent}, "
                f"Urgency: {urgency}, Price sensitivity: {price_sensitivity}"
            )
            
            if deal_status == "closed_won":
                prompt = (
                    f"The customer {lead_name} has agreed to buy the {product_name} at ₹{current_price:,.0f}. "
                    f"Their reply: \"{reply_text}\" indicates they're ready to proceed. "
                    f"Write a warm, professional closing message that:\n"
                    f"1. Thanks them for their decision\n"
                    f"2. Confirms the final price of ₹{current_price:,.0f}\n"
                    f"3. Mentions next steps for delivery/payment\n"
                    f"4. Expresses gratitude and excitement\n"
                    f"Keep it concise and enthusiastic. End with: Best regards, Sasitharan"
                )
                
            elif deal_status == "closed_lost":
                prompt = (
                    f"The customer {lead_name} has declined the {product_name} offer. "
                    f"Their reply: \"{reply_text}\" indicates they're not interested. "
                    f"Write a graceful, professional closing message that:\n"
                    f"1. Thanks them for their time and consideration\n"
                    f"2. Respects their decision\n"
                    f"3. Leaves the door open for future opportunities\n"
                    f"4. Maintains a positive, understanding tone\n"
                    f"Keep it brief and respectful. End with: Best regards, Sasitharan"
                )
                
            else:
                # Ongoing negotiation
                margin_info = f"Maintain at least {((min_price - cost_price) / min_price) * 100:.1f}% profit margin. New offer maintains {new_profit_margin:.1f}% margin."
                
                prompt = (
                    f"You are negotiating with customer {lead_name} for a {product_name}. "
                    f"Customer analysis: {context}. Customer replied: \"{reply_text}\" "
                    f"Current offer: ₹{current_price:,.0f}, New strategic offer: ₹{new_price:,.0f}. "
                    f"{margin_info}\n\n"
                    f"Write a persuasive, contextually appropriate response that:\n"
                    f"1. Acknowledges their specific concern/interest based on their sentiment\n"
                    f"2. Presents the new price naturally and persuasively\n"
                    f"3. Addresses their intent (price concern, need info, etc.)\n"
                    f"4. Creates appropriate urgency without being pushy\n"
                    f"5. Asks for decision or next step\n\n"
                    f"Keep it conversational, empathetic, and professional (2-3 sentences). "
                    f"End with: Best regards, Sasitharan"
                )
            
            client = cohere.Client(COHERE_API_KEY)
            resp = client.chat(model=PRIMARY_MODEL, message=prompt, temperature=0.7)
            
            if resp.text.strip():
                # Update lead negotiation state
                stage = "closed_won" if deal_status == "closed_won" else "closed_lost" if deal_status == "closed_lost" else "negotiating"
                ensure_lead_exists_in_db_only(conn, lead_name, lead_email, product_name, new_price)
                return resp.text.strip(), new_price, deal_status
                
        except Exception as e:
            st.warning(f"Cohere error in negotiation: {e}")
    
    # Fallback messages based on deal status
    if deal_status == "closed_won":
        message = (
            f"Fantastic, {lead_name}! Thank you for choosing the {product_name} at ₹{current_price:,.0f}. "
            f"I'll send you the payment details and delivery information shortly. "
            f"You've made an excellent choice!\n\nBest regards,\nSasitharan"
        )
        
    elif deal_status == "closed_lost":
        message = (
            f"Hi {lead_name}, I completely understand your decision. "
            f"Thank you for taking the time to consider our {product_name}. "
            f"If your needs change in the future, please don't hesitate to reach out. "
            f"We're always here to help!\n\nBest regards,\nSasitharan"
        )
        
    else:
        # Ongoing negotiation fallback
        responses = [
            f"Hi {lead_name}, I appreciate your feedback! Let me offer you ₹{new_price:,.0f} for the {product_name}. This is a great value for the quality you're getting. What do you think?",
            f"Thank you for your interest, {lead_name}! I can work with ₹{new_price:,.0f} for you on this {product_name}. This maintains excellent quality while fitting your budget. Shall we move forward?",
            f"I understand your concerns, {lead_name}. How about ₹{new_price:,.0f} for the {product_name}? This is my best offer while ensuring you get premium quality. Ready to proceed?"
        ]
        message = random.choice(responses) + f"\n\nBest regards,\nSasitharan"
    
    # Update lead negotiation state
    stage = "closed_won" if deal_status == "closed_won" else "closed_lost" if deal_status == "closed_lost" else "negotiating"
    ensure_lead_exists_in_db_only(conn, lead_name, lead_email, product_name, new_price)
    
    return message, new_price, deal_status

# ---------------------------
# ENHANCED REPLY CLASSIFIER (now uses AI analysis)
# ---------------------------
def classify_reply(text: str) -> str:
    """Classify reply using AI sentiment analysis"""
    if not text or text.strip() == "":
        return "ignore"
    
    analysis = analyze_customer_sentiment(text)
    intent = analysis.get("intent", "neutral")
    
    # Map AI intents to classifications
    intent_mapping = {
        "buy_ready": "deal_closed",
        "reject": "reject",
        "price_concern": "price_negotiation",
        "need_info": "interested",
        "schedule_call": "book",
        "neutral": "neutral"
    }
    
    return intent_mapping.get(intent, "neutral")

# ---------------------------
# DEAL QUALITY (enhanced with AI insights)
# ---------------------------
def compute_deal_quality(lead_record: Dict, classification: str, product_name: str = None, ai_analysis: Dict = None) -> float:
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
    
    # AI analysis bonus
    ai_bonus = 0.0
    if ai_analysis:
        confidence = ai_analysis.get("confidence", 0.5)
        urgency = ai_analysis.get("urgency", "medium")
        if urgency == "high":
            ai_bonus = 0.15 * confidence
        elif urgency == "medium":
            ai_bonus = 0.08 * confidence
        else:
            ai_bonus = 0.03 * confidence
    
    return max(0.0, min(1.0, role_weight + prior + class_bonus + product_bonus + ai_bonus))

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
# ENHANCED Auto-refresh inbox function with AI-powered responses
# ---------------------------
def auto_check_inbox():
    """Function to automatically check inbox and process replies with AI negotiation logic"""
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
                
                # Analyze customer sentiment with AI
                ai_analysis = analyze_customer_sentiment(reply_body)
                
                # Classify the reply using AI analysis
                classification = classify_reply(reply_body)
                
                # Store reply for manual processing - NO AUTO-RESPONSE
                current_product = negotiation_state["product"] if negotiation_state else None
                current_price = negotiation_state["price"] if negotiation_state else None
                
                # Determine deal outcome based on AI analysis
                intent = ai_analysis.get("intent", "neutral")
                sentiment = ai_analysis.get("sentiment", "neutral")
                
                if intent == "reject" and sentiment == "negative":
                    # Customer rejected - close the deal
                    reward = -2.0
                    deal_closed = -1  # Mark as rejected
                    negotiation_stage = "closed_lost"
                    
                elif intent == "buy_ready" and sentiment == "positive":
                    # Customer agreed to buy
                    reward = 5.0
                    deal_closed = 1
                    negotiation_stage = "closed_won"
                    
                elif intent in ["price_concern", "need_info", "schedule_call"] and current_product:
                    # Continue negotiation (but don't auto-respond)
                    reward = 1.0
                    deal_closed = 0
                    negotiation_stage = "negotiating"
                        
                else:
                    # Neutral or other responses
                    reward = 0.0
                    deal_closed = 0
                    negotiation_stage = negotiation_state["stage"] if negotiation_state else "initial"
                
                # Compute deal quality with AI insights
                deal_quality = compute_deal_quality(lead, classification, current_product, ai_analysis)
                
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
                
                # Show the processed reply with AI insights
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
                    
                    # Show AI analysis insights
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.info(f"**AI Analysis:**\n- Sentiment: {sentiment.title()}\n- Intent: {intent.replace('_', ' ').title()}")
                    with col_right:
                        if current_product:
                            st.info(f"**Product:** {current_product}\n**Current Price:** ₹{current_price:,.0f}" if current_price else f"**Product:** {current_product}")
                    
                    st.write("**Customer Reply:**")
                    st.write(reply_body[:1000])
                    
                    # Store this reply for manual response generation with AI analysis
                    st.session_state[f"pending_reply_{lead_email}"] = {
                        "lead_name": lead_name,
                        "lead_email": lead_email,
                        "reply_body": reply_body,
                        "classification": classification,
                        "product": current_product,
                        "price": current_price,
                        "subject": rep.get('subject', ''),
                        "ai_analysis": ai_analysis  # Include AI analysis
                    }
        
        return processed
    return 0

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="AI-Powered SDR Simulator — Intelligent Sales Negotiation", layout="wide")
st.title("🤖 AI-Powered SDR Simulator — Intelligent Sales Negotiation with Cohere")

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
    st.header("🧠 AI Features")
    if COHERE_INSTALLED and COHERE_API_KEY:
        st.success("✅ Cohere AI Enabled")
        st.info("• Intelligent sentiment analysis\n• Context-aware responses\n• Smart deal closure detection\n• Adaptive negotiation strategies")
    else:
        st.warning("⚠️ Cohere AI Disabled")
        st.info("Set COHERE_API_KEY environment variable to enable AI features")
    
    st.markdown("---")
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
        if st.button("🎯 Generate & Send AI-Powered Introduction", type="primary"):
            if not lead_email:
                st.error("Please provide the lead's email.")
            else:
                with st.spinner("Generating AI-powered product introduction..."):
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
                with st.spinner("Sending AI-powered introduction..."):
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
                        st.success(f"✅ AI-powered introduction sent for {selected_product}!")
                        
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
    st.header("🧠 AI-Powered Negotiation Center")
    
    # Manual check inbox button
    if st.button("🔍 Manual Inbox Check"):
        with st.spinner("Checking inbox..."):
            processed = auto_check_inbox()
            if processed == 0:
                st.info("No new messages found.")
    
    # Auto-refresh status
    if st.session_state.get("auto_refresh_enabled"):
        st.success("🔄 Auto-refresh is ENABLED (every 10 seconds)")
        st.info("🧠 New replies analyzed with AI - respond intelligently below")
    else:
        st.warning("⏸️ Auto-refresh is DISABLED")
    
    st.markdown("---")
    
    # AI Negotiator Response Section
    st.subheader("🤖 Intelligent AI Responses")
    
    # Check for pending replies that need responses
    pending_replies = []
    for key in st.session_state.keys():
        if key.startswith("pending_reply_") and st.session_state[key]:
            pending_replies.append(st.session_state[key])
    
    if pending_replies:
        st.success(f"📬 {len(pending_replies)} reply(s) ready for intelligent negotiation!")
        
        for reply_data in pending_replies:
            lead_name = reply_data["lead_name"]
            lead_email = reply_data["lead_email"]
            classification = reply_data["classification"]
            product = reply_data["product"]
            current_price = reply_data["price"]
            reply_body = reply_data["reply_body"]
            ai_analysis = reply_data.get("ai_analysis", {})
            
            with st.expander(f"🎯 AI Negotiate with {lead_name} - {classification.upper().replace('_', ' ')}", expanded=True):
                # Show AI analysis and product info
                col_ai, col_product = st.columns(2)
                
                with col_ai:
                    st.info(f"**🧠 AI Analysis:**\n- Sentiment: {ai_analysis.get('sentiment', 'unknown').title()}\n- Intent: {ai_analysis.get('intent', 'unknown').replace('_', ' ').title()}\n- Urgency: {ai_analysis.get('urgency', 'unknown').title()}\n- Price Sensitivity: {ai_analysis.get('price_sensitivity', 'unknown').title()}")
                
                with col_product:
                    st.write(f"**Product:** {product}")
                    if current_price:
                        # Calculate profit margin info
                        if product and product in PRODUCTS:
                            min_price = PRODUCTS[product]["min_price"]
                            cost_price = min_price * 0.7
                            profit_margin = ((current_price - cost_price) / current_price) * 100
                            st.info(f"**Current Price:** ₹{current_price:,.0f}\n**Profit Margin:** {profit_margin:.1f}%")
                        else:
                            st.info(f"**Current Price:** ₹{current_price:,.0f}")
                
                st.write("**Customer Reply:**")
                st.write(f'"{reply_body[:300]}..."' if len(reply_body) > 300 else f'"{reply_body}"')
                
                # Generate intelligent AI response
                col_gen, col_send = st.columns(2)
                
                with col_gen:
                    if st.button(f"🧠 Generate Intelligent AI Response", key=f"gen_{lead_email}"):
                        if product and current_price:
                            with st.spinner("Generating intelligent AI response..."):
                                try:
                                    response_msg, new_price, deal_status = generate_intelligent_negotiator_message(
                                        lead_name, reply_body, product, current_price, lead_email
                                    )
                                    
                                    # Store the generated response with deal status
                                    st.session_state[f"generated_response_{lead_email}"] = {
                                        "message": response_msg,
                                        "new_price": new_price,
                                        "deal_status": deal_status,
                                        "subject": f"Re: {reply_data.get('subject', 'Our conversation')}",
                                        "ai_analysis": ai_analysis
                                    }
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating AI response: {e}")
                        else:
                            st.error("Missing product or price information")
                
                # Show generated response if available
                if f"generated_response_{lead_email}" in st.session_state:
                    response_data = st.session_state[f"generated_response_{lead_email}"]
                    deal_status = response_data.get("deal_status", "ongoing")
                    
                    # Show deal status indicator
                    if deal_status == "closed_won":
                        st.success("🎉 **DEAL WON** - Customer ready to buy!")
                    elif deal_status == "closed_lost":
                        st.error("❌ **DEAL LOST** - Customer not interested")
                    else:
                        st.info("🔄 **ONGOING** - Continue negotiation")
                    
                    st.write("**Generated AI Response:**")
                    st.text_area("AI Response:", value=response_data["message"], height=150, key=f"preview_{lead_email}")
                    
                    # Show price change
                    if current_price and response_data["new_price"] != current_price:
                        price_change = response_data["new_price"] - current_price
                        price_color = "red" if price_change < 0 else "green"
                        st.markdown(f"**Price Change:** <span style='color:{price_color}'>₹{price_change:+,.0f}</span> (New: ₹{response_data['new_price']:,.0f})", unsafe_allow_html=True)
                    
                    with col_send:
                        if st.button(f"📤 Send AI Response", key=f"send_{lead_email}", type="primary"):
                            with st.spinner("Sending intelligent AI response..."):
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
                                    st.success("✅ Intelligent AI response sent!")
                                    
                                    # Determine deal closure status for logging
                                    deal_closed_value = 0
                                    if deal_status == "closed_won":
                                        deal_closed_value = 1
                                    elif deal_status == "closed_lost":
                                        deal_closed_value = -1
                                    
                                    # Log the sent response
                                    ts = time.time()
                                    c.execute(
                                        """INSERT INTO logs (ts, lead_name, channel, agent_message, human_reply, classification, 
                                           reward, deal_closed, deal_quality, product_name, current_price, negotiation_stage) 
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                        (ts, lead_name, "Email", response_data["message"], "", "ai_negotiation_response", 
                                         0.0, deal_closed_value, 0.0, product, response_data["new_price"], deal_status),
                                    )
                                    conn.commit()
                                    
                                    # Update lead status in database
                                    if deal_status in ["closed_won", "closed_lost"]:
                                        cursor = conn.cursor()
                                        cursor.execute(
                                            "UPDATE leads SET negotiation_stage = ? WHERE email = ?",
                                            (deal_status, lead_email)
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
        st.info("📭 No pending replies. New customer replies will appear here for AI-powered negotiation.")
    
    # Quick-reply testing section
    st.markdown("---")
    st.markdown("**🧪 Quick-reply examples (for testing AI negotiation):**")
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("✅ Accept Deal"): 
            st.session_state["pending_reply"] = "Yes, I'll take it at that price. Let's proceed with the purchase!"
            st.rerun()
        if st.button("💬 Show Interest"): 
            st.session_state["pending_reply"] = "This looks interesting. Can you tell me more about the features and warranty?"
            st.rerun()
        if st.button("📅 Book Meeting"): 
            st.session_state["pending_reply"] = "I'm interested but would like to schedule a call to discuss this in detail."
            st.rerun()
    
    with col_b:
        if st.button("💸 Price Objection"): 
            st.session_state["pending_reply"] = "The price seems too high for my budget. Can you do better? I was thinking around ₹50,000."
            st.rerun()
        if st.button("❌ Reject"): 
            st.session_state["pending_reply"] = "Thanks, but I'm not interested in this product right now."
            st.rerun()
        if st.button("🤔 Need Time"): 
            st.session_state["pending_reply"] = "I need to think about this and discuss with my team first."
            st.rerun()

# ---------------------------
# ENHANCED METRICS WITH AI INSIGHTS
# ---------------------------
st.markdown("---")
st.header("📊 Enhanced AI-Powered Sales Analytics")

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

# AI-powered metrics
ai_responses = len(logs_df[logs_df["classification"].str.contains("ai_", na=False)]) if total_interactions > 0 else 0
intelligent_closes = len(logs_df[(logs_df["deal_closed"] == 1) & (logs_df["classification"].str.contains("ai_", na=False))]) if total_interactions > 0 else 0

colA, colB, colC, colD, colE, colF, colG = st.columns(7)
colA.metric("Total Interactions", total_interactions)
colB.metric("Deals Closed", deals_closed)
colC.metric("Deals Rejected", deals_rejected)
colD.metric("Success Rate", f"{success_rate:.1f}%")
colE.metric("AI Responses", ai_responses)
colF.metric("Total Revenue", f"₹{total_revenue:,.0f}")
colG.metric("Total Profit", f"₹{total_profit:,.0f}")

# Enhanced analytics with AI insights
if total_interactions > 0:
    st.subheader("📈 AI-Enhanced Analytics Dashboard")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("**Classification Breakdown:**")
        classification_breakdown = logs_df["classification"].value_counts().rename_axis("classification").reset_index(name="count")
        st.bar_chart(classification_breakdown.set_index("classification"))
        
        # AI vs Manual response comparison
        if ai_responses > 0:
            st.write("**AI vs Manual Responses:**")
            ai_vs_manual = {
                "AI Responses": ai_responses,
                "Manual Responses": total_interactions - ai_responses
            }
            ai_df = pd.DataFrame(list(ai_vs_manual.items()), columns=["Type", "Count"])
            st.bar_chart(ai_df.set_index("Type"))
    
    with col_right:
        st.write("**Product Performance:**")
        product_logs = logs_df[logs_df["product_name"].notna()]
        if not product_logs.empty:
            product_breakdown = product_logs["product_name"].value_counts().rename_axis("product").reset_index(name="interactions")
            st.bar_chart(product_breakdown.set_index("product"))
        
        # Deal closure stages
        st.write("**Negotiation Stages:**")
        stage_logs = logs_df[logs_df["negotiation_stage"].notna()]
        if not stage_logs.empty:
            stage_breakdown = stage_logs["negotiation_stage"].value_counts().rename_axis("stage").reset_index(name="count")
            st.bar_chart(stage_breakdown.set_index("stage"))
    
    # Enhanced Profit Margin Analysis with AI insights
    st.subheader("💰 AI-Enhanced Profit Analysis")
    if not closed_deals.empty:
        profit_data = []
        for _, deal in closed_deals.iterrows():
            if deal["product_name"] in PRODUCTS:
                min_price = PRODUCTS[deal["product_name"]]["min_price"]
                cost_price = min_price * 0.7
                profit = deal["current_price"] - cost_price
                profit_margin = (profit / deal["current_price"]) * 100
                is_ai_deal = "ai_" in str(deal["classification"]).lower()
                profit_data.append({
                    "Product": deal["product_name"],
                    "Sale Price": deal["current_price"],
                    "Profit": profit,
                    "Margin %": profit_margin,
                    "Lead": deal["lead_name"],
                    "AI Assisted": "Yes" if is_ai_deal else "No",
                    "Stage": deal.get("negotiation_stage", "unknown")
                })
        
        if profit_data:
            profit_df = pd.DataFrame(profit_data)
            st.dataframe(profit_df, use_container_width=True)
            
            # Compare AI vs Manual deal performance
            if ai_responses > 0:
                col_ai_perf, col_manual_perf = st.columns(2)
                
                with col_ai_perf:
                    ai_deals = profit_df[profit_df["AI Assisted"] == "Yes"]
                    if not ai_deals.empty:
                        st.write("**AI-Assisted Deal Performance:**")
                        st.metric("Avg AI Deal Value", f"₹{ai_deals['Sale Price'].mean():,.0f}")
                        st.metric("Avg AI Profit Margin", f"{ai_deals['Margin %'].mean():.1f}%")
                
                with col_manual_perf:
                    manual_deals = profit_df[profit_df["AI Assisted"] == "No"]
                    if not manual_deals.empty:
                        st.write("**Manual Deal Performance:**")
                        st.metric("Avg Manual Deal Value", f"₹{manual_deals['Sale Price'].mean():,.0f}")
                        st.metric("Avg Manual Profit Margin", f"{manual_deals['Margin %'].mean():.1f}%")
            
            # Average profit margin by product
            avg_margins = profit_df.groupby("Product")["Margin %"].mean().reset_index()
            st.write("**Average Profit Margins by Product:**")
            st.bar_chart(avg_margins.set_index("Product"))
    else:
        st.info("No closed deals yet to analyze profit margins.")
    
    # Recent interactions table with AI insights
    st.subheader("📋 Recent Interactions (AI-Enhanced)")
    display_df = logs_df.copy()
    display_df["time"] = pd.to_datetime(display_df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["AI_Assisted"] = display_df["classification"].str.contains("ai_", na=False)
    
    # Select relevant columns for display
    display_columns = ["time", "lead_name", "classification", "product_name", "current_price", "negotiation_stage", "deal_closed", "AI_Assisted"]
    available_columns = [col for col in display_columns if col in display_df.columns]
    
    st.dataframe(display_df[available_columns].head(50), use_container_width=True)
    
    # AI Performance Summary
    if ai_responses > 0:
        st.subheader("🧠 AI Performance Summary")
        ai_success_rate = (intelligent_closes / ai_responses * 100) if ai_responses > 0 else 0
        
        col_ai_summary = st.columns(3)
        with col_ai_summary[0]:
            st.metric("AI Response Rate", f"{(ai_responses/total_interactions*100):.1f}%")
        with col_ai_summary[1]:
            st.metric("AI Success Rate", f"{ai_success_rate:.1f}%")
        with col_ai_summary[2]:
            st.metric("AI Deal Closes", intelligent_closes)
        
        if COHERE_INSTALLED and COHERE_API_KEY:
            st.success("🧠 Cohere AI is actively enhancing your sales conversations with intelligent sentiment analysis and contextual responses!")
        else:
            st.warning("⚠️ Enable Cohere AI for even better performance with intelligent sentiment analysis and adaptive responses.")
else:
    st.info("No interactions yet. Send some AI-powered product introductions to get started!")

# Add footer with AI capabilities
st.markdown("---")
st.markdown("### 🚀 AI-Powered Features")
if COHERE_INSTALLED and COHERE_API_KEY:
    st.success(
        "✅ **Active AI Features:**\n"
        "• Intelligent sentiment analysis of customer replies\n"
        "• Context-aware response generation\n"
        "• Automatic deal closure detection\n"
        "• Adaptive pricing strategies\n"
        "• Smart negotiation tactics based on customer intent"
    )
else:
    st.error(
        "❌ **AI Features Disabled:**\n"
        "• Set COHERE_API_KEY environment variable to enable\n"
        "• Get your API key from: https://dashboard.cohere.com/\n"
        "• Restart the application after setting the key"
    )