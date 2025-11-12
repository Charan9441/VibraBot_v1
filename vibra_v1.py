import streamlit as st
import sqlite3
import time
import json
from openai import OpenAI
from typing import List, Dict
import hashlib

# Optional: SerpAPI for web search
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

# ---------------------------
# CONFIG & API SETUP
# ---------------------------
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

serpapi_key = None
if "SERPAPI_KEY" in st.secrets:
    serpapi_key = st.secrets["SERPAPI_KEY"]

client = None
if api_key:
    client = OpenAI(api_key=api_key)

if not api_key:
    st.error("❌ OpenAI API key not found. Set OPENAI_API_KEY in ~/.streamlit/secrets.toml")
    st.stop()

# ---------------------------
# SQLITE DATABASE - ENHANCED FOR LEARNING
# ---------------------------
DB_PATH = "vibra_learning.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Conversations table
c.execute(
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_id TEXT UNIQUE,
        title TEXT,
        created_ts REAL,
        updated_ts REAL
    )
    """
)

# Chat messages table
c.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_id TEXT,
        role TEXT,
        text TEXT,
        ts REAL
    )
    """
)

# Learning database - stores Q&A pairs
c.execute(
    """
    CREATE TABLE IF NOT EXISTS learned_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_hash TEXT UNIQUE,
        question TEXT,
        answer TEXT,
        confidence REAL DEFAULT 0.5,
        usage_count INTEGER DEFAULT 1,
        created_ts REAL,
        updated_ts REAL
    )
    """
)

# User feedback for learning
c.execute(
    """
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        response_id INTEGER,
        feedback TEXT,
        rating INTEGER,
        ts REAL
    )
    """
)

# Knowledge base - custom facts learned
c.execute(
    """
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        fact TEXT,
        learned_from TEXT,
        ts REAL
    )
    """
)

conn.commit()


def hash_question(question: str) -> str:
    """Create a hash of the question for quick lookup"""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def save_learned_response(question: str, answer: str):
    """Save Q&A pair to learning database"""
    q_hash = hash_question(question)
    try:
        c.execute(
            "INSERT INTO learned_responses (question_hash, question, answer, created_ts, updated_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (q_hash, question, answer, time.time(), time.time())
        )
    except sqlite3.IntegrityError:
        # Update if exists
        c.execute(
            "UPDATE learned_responses SET usage_count = usage_count + 1, updated_ts = ? WHERE question_hash = ?",
            (time.time(), q_hash)
        )
    conn.commit()


def get_learned_response(question: str) -> Dict:
    """Retrieve learned response if exists"""
    q_hash = hash_question(question)
    c.execute("SELECT answer, confidence, usage_count FROM learned_responses WHERE question_hash = ?", (q_hash,))
    row = c.fetchone()
    if row:
        return {"answer": row[0], "confidence": row[1], "usage_count": row[2]}
    return None


def update_response_confidence(question: str, new_confidence: float):
    """Update confidence score based on user feedback"""
    q_hash = hash_question(question)
    c.execute(
        "UPDATE learned_responses SET confidence = ? WHERE question_hash = ?",
        (new_confidence, q_hash)
    )
    conn.commit()


def save_user_feedback(response_id: int, feedback: str, rating: int):
    """Save user feedback for learning"""
    c.execute(
        "INSERT INTO user_feedback (response_id, feedback, rating, ts) VALUES (?, ?, ?, ?)",
        (response_id, feedback, rating, time.time())
    )
    conn.commit()


def add_to_knowledge_base(topic: str, fact: str, source: str = "user"):
    """Add learned fact to knowledge base"""
    c.execute(
        "INSERT INTO knowledge_base (topic, fact, learned_from, ts) VALUES (?, ?, ?, ?)",
        (topic, fact, source, time.time())
    )
    conn.commit()


def get_knowledge_base(topic: str = None) -> List[Dict]:
    """Retrieve knowledge base facts"""
    if topic:
        c.execute("SELECT topic, fact, learned_from FROM knowledge_base WHERE topic LIKE ? ORDER BY ts DESC", (f"%{topic}%",))
    else:
        c.execute("SELECT topic, fact, learned_from FROM knowledge_base ORDER BY ts DESC LIMIT 20")
    rows = c.fetchall()
    return [{"topic": r[0], "fact": r[1], "source": r[2]} for r in rows]


def save_conversation_message(conv_id: str, role: str, text: str):
    """Save message to conversation"""
    c.execute(
        "INSERT INTO chat_messages (conv_id, role, text, ts) VALUES (?, ?, ?, ?)",
        (conv_id, role, text, time.time())
    )
    c.execute(
        "UPDATE conversations SET updated_ts = ? WHERE conv_id = ?",
        (time.time(), conv_id)
    )
    conn.commit()


def load_conversation_messages(conv_id: str) -> List[Dict]:
    """Load all messages from a conversation"""
    c.execute(
        "SELECT role, text FROM chat_messages WHERE conv_id = ? ORDER BY ts ASC",
        (conv_id,)
    )
    rows = c.fetchall()
    return [{"role": r[0], "text": r[1]} for r in rows]


def create_conversation(conv_id: str, title: str):
    """Create a new conversation"""
    try:
        c.execute(
            "INSERT INTO conversations (conv_id, title, created_ts, updated_ts) VALUES (?, ?, ?, ?)",
            (conv_id, title, time.time(), time.time())
        )
        conn.commit()
    except:
        pass


def get_all_conversations() -> List[Dict]:
    """Get all conversations"""
    c.execute("SELECT conv_id, title, updated_ts FROM conversations ORDER BY updated_ts DESC")
    rows = c.fetchall()
    return [{"id": r[0], "title": r[1], "updated": r[2]} for r in rows]


def delete_conversation(conv_id: str):
    """Delete a conversation"""
    c.execute("DELETE FROM chat_messages WHERE conv_id = ?", (conv_id,))
    c.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
    conn.commit()


# ---------------------------
# PAGE CONFIG & STYLING
# ---------------------------
st.set_page_config(page_title="VibraBot - AI Learning", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0 !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1a1a3e !important;
        color: #00ff88 !important;
        border: 2px solid #00ff88 !important;
    }
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: white;
        border-left: 4px solid #00ff88;
    }
    .chat-message-bot {
        background: linear-gradient(135deg, #1a1a3e 0%, #2d2d5f 100%);
        border-radius: 15px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #00ff88;
        border-left: 4px solid #667eea;
    }
    .learning-badge {
        background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.8em;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# TEXT ANALYSIS
# ---------------------------
def analyze_text(text: str) -> Dict:
    """Analyze text"""
    text = text or ""
    vowels_count = sum(1 for ch in text.lower() if ch in "aeiou")
    digits_count = sum(1 for ch in text if ch.isdigit())
    length = len(text)
    vowel_percent = (vowels_count * 100 / length) if length else 0.0
    digit_percent = (digits_count * 100 / length) if length else 0.0
    
    return {
        "original": text,
        "reversed": text[::-1],
        "length": length,
        "vowels": vowels_count,
        "digits": digits_count,
        "vowel_percent": vowel_percent,
        "digit_percent": digit_percent,
    }


# ---------------------------
# WEB SEARCH
# ---------------------------
def web_search(query: str, num: int = 3) -> List[Dict]:
    """Search web using SerpAPI"""
    if not SERPAPI_AVAILABLE or not serpapi_key:
        return []
    try:
        params = {"q": query, "engine": "google", "num": num, "api_key": serpapi_key}
        search = GoogleSearch(params)
        results = search.get_dict()
        hits = []
        for r in results.get("organic_results", [])[:num]:
            hits.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", "")
            })
        return hits
    except:
        return []


# ---------------------------
# OPENAI API
# ---------------------------
def call_openai_chat(messages: List[Dict], model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Call OpenAI API - optimized for speed"""
    if not client:
        raise RuntimeError("OpenAI client not configured.")
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9  # Faster generation
    )
    return resp.choices[0].message.content


def get_response_with_learning(user_message: str, history: List[Dict], use_learned: bool = True) -> str:
    """Get AI response with learning database integration"""
    
    # Check if we've learned this before
    if use_learned:
        learned = get_learned_response(user_message)
        if learned and learned["confidence"] > 0.8:
            return f"📚 *[Learned Response]* {learned['answer']}"
    
    # Get knowledge base context
    kb = get_knowledge_base()
    kb_context = ""
    if kb:
        kb_context = "Knowledge learned so far:\n" + "\n".join([f"- {k['topic']}: {k['fact']}" for k in kb[:5]])
    
    system_prompt = (
        "You are VibraBot, an advanced AI that learns from conversations. "
        "You remember previous interactions and improve over time. "
        "You are helpful, creative, and adaptive. Use markdown for formatting. "
        f"{kb_context}\n"
        "Learn new facts when shared by the user and incorporate them into future responses."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (last 10 messages) - convert 'bot' to 'assistant'
    for msg in history[-10:]:
        role = "assistant" if msg["role"] == "bot" else "user"
        messages.append({"role": role, "content": msg["text"]})
    
    messages.append({"role": "user", "content": user_message})
    
    return call_openai_chat(messages, temperature=0.7, max_tokens=2000)


# ---------------------------
# SESSION STATE
# ---------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(int(time.time() * 1000))

if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = st.session_state.session_id
    create_conversation(st.session_state.current_conv_id, "Learning Conversation")

if "chat" not in st.session_state:
    msgs = load_conversation_messages(st.session_state.current_conv_id)
    if msgs:
        st.session_state.chat = msgs
    else:
        st.session_state.chat = [{"role": "bot", "text": "🧠 Hi! I'm VibraBot with AI learning capabilities. I remember what you teach me!"}]

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "gpt-4o-mini"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "learning_enabled" not in st.session_state:
    st.session_state.learning_enabled = True


# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("🧠 VibraBot AI Learning")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕ New"):
            st.session_state.current_conv_id = str(int(time.time() * 1000))
            create_conversation(st.session_state.current_conv_id, "New Learning Session")
            st.session_state.chat = [{"role": "bot", "text": "🧠 New learning session started!"}]
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.chat = [{"role": "bot", "text": "✨ Conversation cleared!"}]
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    st.markdown("---")
    
    st.subheader("💬 Conversations")
    conversations = get_all_conversations()
    for conv in conversations[:8]:
        col_a, col_b = st.columns([4, 1])
        with col_a:
            if st.button(f"📝 {conv['title'][:20]}", key=conv["id"]):
                st.session_state.current_conv_id = conv["id"]
                st.session_state.chat = load_conversation_messages(conv["id"])
                st.rerun()
        with col_b:
            if st.button("🗑", key=f"del_{conv['id']}", help="Delete"):
                delete_conversation(conv["id"])
                st.rerun()
    
    st.markdown("---")
    
    st.subheader("⚙️ Settings")
    
    st.session_state.learning_enabled = st.checkbox("🧠 Enable Learning", value=True)
    
    st.session_state.model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )
    
    st.markdown("---")
    
    st.subheader("📚 Knowledge Base")
    kb_items = get_knowledge_base()
    if kb_items:
        for item in kb_items[:5]:
            st.info(f"**{item['topic']}**: {item['fact'][:60]}...")
    else:
        st.write("No learned facts yet")
    
    st.markdown("---")
    
    st.subheader("📊 Learning Stats")
    c.execute("SELECT COUNT(*) FROM learned_responses WHERE confidence > 0.8")
    high_conf = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM knowledge_base")
    kb_count = c.fetchone()[0]
    st.metric("High Confidence Responses", high_conf)
    st.metric("Learned Facts", kb_count)
    
    st.markdown("---")
    st.caption("🧠 Powered by Auto-Learning AI | Made by Charan")


# ---------------------------
# MAIN CONTENT
# ---------------------------
st.title("🧠 VibraBot - Auto-Learning AI")
st.markdown("*An AI that learns and improves from every conversation*")
st.markdown("---")

# Chat display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat:
        role = msg["role"]
        if role == "bot":
            st.markdown(f"<div class='chat-message-bot'>🤖 **VibraBot:** {msg['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message-user'>👤 **You:** {msg['text']}</div>", unsafe_allow_html=True)

st.markdown("---")

# Input
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_area(
        "Type your message...",
        height=80,
        placeholder="Ask anything! The more you teach me, the smarter I get..."
    )
with col2:
    st.write("")
    send_button = st.button("📤 Send", use_container_width=True)

if send_button and user_input:
    msg = user_input.strip()
    lower = msg.lower()
    
    # ANALYZE
    if lower.startswith("analyze:"):
        payload = msg.split(":", 1)[1].strip()
        if payload:
            res = analyze_text(payload)
            reply = (
                f"📊 **Analysis:**\n"
                f"• Original: `{res['original']}`\n"
                f"• Length: {res['length']}\n"
                f"• Vowels: {res['vowels']} ({res['vowel_percent']:.1f}%)\n"
                f"• Digits: {res['digits']} ({res['digit_percent']:.1f}%)\n"
                f"• Reversed: `{res['reversed']}`"
            )
        else:
            reply = "❌ Usage: analyze: your text"
    
    # REVERSE
    elif lower.startswith("reverse:"):
        payload = msg.split(":", 1)[1].strip()
        reply = f"🔄 **Reversed:** `{payload[::-1]}`" if payload else "❌ Usage: reverse: your text"
    
    # SEARCH
    elif lower.startswith("search:"):
        query = msg.split(":", 1)[1].strip()
        hits = web_search(query, num=3)
        if hits:
            search_text = "\n".join([f"[{i+1}] {h['title']}: {h['snippet']}" for i, h in enumerate(hits)])
            reply = get_response_with_learning(msg, st.session_state.chat, use_learned=st.session_state.learning_enabled)
        else:
            reply = "❌ Web search unavailable"
    
    # TEACH (add to knowledge base)
    elif lower.startswith("teach:"):
        info = msg.split(":", 1)[1].strip()
        parts = info.split("|")
        if len(parts) == 2:
            topic, fact = parts[0].strip(), parts[1].strip()
            add_to_knowledge_base(topic, fact, "user")
            reply = f"✅ **Learned!** Topic: {topic}\nFact: {fact}"
        else:
            reply = "❌ Usage: teach: topic | fact"
    
    # REGULAR CHAT
    else:
        reply = get_response_with_learning(msg, st.session_state.chat, use_learned=st.session_state.learning_enabled)
        
        # Auto-save to learning database
        if st.session_state.learning_enabled:
            save_learned_response(msg, reply)
    
    # Add to chat
    st.session_state.chat.append({"role": "user", "text": user_input})
    st.session_state.chat.append({"role": "bot", "text": reply})
    
    # Save to database
    save_conversation_message(st.session_state.current_conv_id, "user", user_input)
    save_conversation_message(st.session_state.current_conv_id, "bot", reply)
    
    st.rerun()


# Footer
st.markdown("---")
st.caption("""
💡 **Commands:**
• `analyze: text` - Analyze text
• `reverse: text` - Reverse text  
• `search: query` - Web search
• `teach: topic | fact` - Teach me new facts
• Regular chat - I'll learn and remember!
""")