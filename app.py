"""
app.py - ZenShine Pro AI Chatbot
RAG (Qdrant) + Groq LLM — premium UI
"""

import os
import warnings
import httpx
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from rag import ingest_pdf, retrieve, collection_count

load_dotenv()

warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ZenShine Pro — AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400;1,600&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-dark:   #1a3a2a;
    --green-mid:    #2d5a3d;
    --green-soft:   #3d7a52;
    --green-pale:   #e8f5ee;
    --green-accent: #4caf70;
    --white:        #ffffff;
    --off-white:    #f7faf8;
    --cream:        #f2f7f4;
    --border:       #ddeae3;
    --border-soft:  #eaf3ee;
    --text-dark:    #0f2218;
    --text-mid:     #3d5c48;
    --text-muted:   #7a9986;
    --shadow-sm:    0 1px 4px rgba(26,58,42,0.07);
    --shadow-md:    0 4px 20px rgba(26,58,42,0.10);
    --shadow-lg:    0 8px 40px rgba(26,58,42,0.13);
    --radius-xl:    20px;
    --radius-lg:    16px;
    --radius-md:    12px;
    --radius-sm:    8px;
    --radius-pill:  100px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 14px;
    color: var(--text-dark);
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: var(--off-white);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 1.4rem 2rem 1rem 3.8rem !important;
    max-width: 100% !important;
}

/* ── Sidebar toggle — pin to top-left, never overlap main content ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 1rem !important;
    left: 1rem !important;
    z-index: 999 !important;
}

/* ── Sidebar toggle — always visible, even when collapsed ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
section[data-testid="stSidebar"] ~ div button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* Style the toggle button itself */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    background: var(--green-dark) !important;
    border: 1.5px solid rgba(76,175,112,0.4) !important;
    border-radius: 10px !important;
    width: 36px !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 2px 2px 14px rgba(0,0,0,0.22) !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: var(--green-mid) !important;
    border-color: rgba(76,175,112,0.7) !important;
}
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg {
    fill: #7ecf9a !important;
    stroke: #7ecf9a !important;
    color: #7ecf9a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
    border-right: 1px solid rgba(77,175,112,0.15) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.12) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

[data-testid="stSidebar"] * { color: #c8ddd0 !important; }

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(77,175,112,0.2) !important;
    color: #c8ddd0 !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.78rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    text-transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(76,175,112,0.12) !important;
    border-color: rgba(76,175,112,0.4) !important;
    color: #a8e8c0 !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(77,175,112,0.1) !important;
    margin: 1.1rem 0 !important;
}

[data-testid="stSidebar"] .stFileUploader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px dashed rgba(77,175,112,0.22) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5rem !important;
}

[data-testid="stSidebar"] .stSuccess > div {
    background: rgba(76,175,112,0.12) !important;
    border: 1px solid rgba(76,175,112,0.25) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.76rem !important;
}
[data-testid="stSidebar"] .stWarning > div {
    background: rgba(230,180,60,0.10) !important;
    border: 1px solid rgba(230,180,60,0.22) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.76rem !important;
}
[data-testid="stSidebar"] .stError > div {
    background: rgba(220,60,60,0.10) !important;
    border: 1px solid rgba(220,60,60,0.22) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.76rem !important;
}

/* ── Sidebar brand block ── */
.sb-brand {
    padding: 1.8rem 1.4rem 1.5rem;
    border-bottom: 1px solid rgba(77,175,112,0.12);
    background: linear-gradient(180deg, rgba(76,175,112,0.07) 0%, transparent 100%);
}
.sb-logo-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}
.sb-logo-box {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark));
    border: 1.5px solid rgba(76,175,112,0.35);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    flex-shrink: 0;
}
.sb-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #eef6f1 !important;
    line-height: 1.1;
    letter-spacing: -0.01em;
}
.sb-wordmark span { color: #7ecf9a !important; font-style: italic; }
.sb-tagline {
    font-size: 0.68rem;
    color: rgba(160,200,175,0.65) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 3px;
}
.sb-live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.38rem 0.75rem;
    background: rgba(76,175,112,0.08);
    border: 1px solid rgba(76,175,112,0.15);
    border-radius: 100px;
    margin-top: 10px;
}
.sb-live-dot {
    width: 6px; height: 6px;
    background: #5fdc88;
    border-radius: 50%;
    animation: livepulse 2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes livepulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.3; transform: scale(0.65); }
}
.sb-live-text {
    font-size: 0.66rem;
    color: rgba(160,210,180,0.75) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

/* Sidebar section heading */
.sb-section {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 1.4rem 0 0.75rem;
}
.sb-section-line { flex: 1; height: 1px; background: rgba(77,175,112,0.12); }
.sb-section-label {
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: rgba(76,175,112,0.5) !important;
    white-space: nowrap;
}

/* Status card */
.status-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(77,175,112,0.15);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
}
.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.32rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.status-row:last-child { border-bottom: none; }
.status-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.73rem;
    color: #8aab96 !important;
}
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.status-indicator.connected { color: #7ecf9a !important; }
.status-indicator.disconnected { color: #e07070 !important; }
.status-dot-ok {
    width: 6px; height: 6px;
    background: #5fdc88;
    border-radius: 50%;
}
.status-dot-err {
    width: 6px; height: 6px;
    background: #e07070;
    border-radius: 50%;
}

/* Services */
.svc-list { display: flex; flex-direction: column; gap: 1px; }
.svc-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0.65rem;
    border-radius: var(--radius-sm);
    transition: background 0.15s;
}
.svc-item:hover { background: rgba(255,255,255,0.05); }
.svc-name { font-size: 0.74rem; color: #9ab8a6 !important; font-weight: 400; }
.svc-price {
    font-size: 0.78rem;
    font-weight: 600;
    color: #7ecf9a !important;
    font-family: 'Playfair Display', serif;
    font-style: italic;
}

/* Discounts */
.disc-list { display: flex; flex-direction: column; gap: 4px; }
.disc-item {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 0.38rem 0.65rem;
    background: rgba(76,175,112,0.05);
    border: 1px solid rgba(76,175,112,0.1);
    border-radius: var(--radius-sm);
}
.disc-dot { width: 5px; height: 5px; background: var(--green-accent); border-radius: 50%; flex-shrink: 0; opacity: 0.6; }
.disc-text { font-size: 0.72rem; color: #9ab8a6 !important; line-height: 1.45; }
.disc-text b { color: #7ecf9a !important; font-weight: 600; }

/* Contact card */
.contact-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(77,175,112,0.12);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    margin-top: 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 0;
}
.contact-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.32rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.contact-row:last-child { border-bottom: none; }
.contact-ico { margin-top: 2px; flex-shrink: 0; }
.contact-detail { font-size: 0.72rem; color: #8aab96 !important; line-height: 1.45; }
.contact-detail b { color: #cce8d8 !important; font-weight: 500; display: block; }

/* ══════════════════════ MAIN INPUT ══════════════════════ */
.stTextInput input {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text-dark) !important;
    font-size: 0.9rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.8rem 1.1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTextInput input:focus {
    border-color: var(--green-soft) !important;
    box-shadow: 0 0 0 3px rgba(61,122,82,0.12) !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: #b0c8bc !important; }

/* ══════════════════════ SEND BUTTON ══════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid) 0%, var(--green-dark) 100%) !important;
    color: #e8f5ee !important;
    border: none !important;
    border-radius: var(--radius-lg) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.8rem 1.5rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: all 0.22s !important;
    white-space: nowrap !important;
    box-shadow: 0 3px 14px rgba(26,58,42,0.22) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--green-soft) 0%, var(--green-mid) 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(26,58,42,0.28) !important;
}

.stSpinner > div { border-top-color: var(--green-soft) !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #b8ccbf; }

/* ══════════════════════ MAIN HEADER ══════════════════════ */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1.5px solid var(--border);
    position: relative;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -1.5px; left: 0;
    width: 72px; height: 2px;
    background: linear-gradient(90deg, var(--green-soft), transparent);
    border-radius: 2px;
}
.header-left { display: flex; align-items: center; gap: 16px; }
.header-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--green-mid) 0%, var(--green-dark) 100%);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: var(--shadow-md);
    flex-shrink: 0;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-dark);
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.main-title em { font-style: italic; color: var(--green-soft); }
.main-subtitle {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 400;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--green-pale);
    border: 1px solid var(--border);
    padding: 0.45rem 1rem;
    border-radius: 100px;
}
.hb-dot { width: 7px; height: 7px; background: var(--green-accent); border-radius: 50%; animation: livepulse 2s ease-in-out infinite; }
.hb-text { font-size: 0.68rem; color: var(--text-mid); letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600; }

/* ══════════════════════ CHAT CONTAINER ══════════════════════ */
.chat-container {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.4rem 1.6rem;
    height: calc(100vh - 290px);
    min-height: 260px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
    margin-bottom: 0.5rem;
    box-shadow: var(--shadow-md);
}

.bubble-row-user { display: flex; justify-content: flex-end; }
.bubble-user {
    background: linear-gradient(135deg, var(--green-mid) 0%, var(--green-dark) 100%);
    color: #e8f5ee;
    padding: 0.82rem 1.15rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 62%;
    font-size: 0.875rem;
    line-height: 1.65;
    font-family: 'Plus Jakarta Sans', sans-serif;
    box-shadow: 0 3px 14px rgba(26,58,42,0.18);
}

.bubble-row-bot { display: flex; justify-content: flex-start; align-items: flex-start; gap: 11px; }

.bot-avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    flex-shrink: 0;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(26,58,42,0.14);
    margin-top: 2px;
    border: 2px solid var(--border);
}

.bubble-bot {
    background: var(--cream);
    border: 1.5px solid var(--border-soft);
    color: var(--text-dark);
    padding: 0.82rem 1.15rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 70%;
    font-size: 0.875rem;
    line-height: 1.75;
    font-family: 'Plus Jakarta Sans', sans-serif;
    box-shadow: var(--shadow-sm);
}
.bubble-bot b, .bubble-bot strong { color: var(--green-dark); font-weight: 600; }
.bubble-bot ul { margin: 0.4rem 0 0.4rem 1.1rem; padding: 0; }
.bubble-bot li { margin-bottom: 0.3rem; }

.welcome-bubble {
    background: linear-gradient(145deg, var(--green-dark) 0%, var(--green-mid) 100%);
    border: 1px solid rgba(76,175,112,0.2);
    color: #cce0d4;
    padding: 1.3rem 1.5rem;
    border-radius: 4px 20px 20px 20px;
    max-width: 72%;
    font-size: 0.875rem;
    line-height: 1.75;
    font-family: 'Plus Jakarta Sans', sans-serif;
    box-shadow: 0 6px 28px rgba(26,58,42,0.18);
}
.wlc-greeting {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #eaf5ee !important;
    margin-bottom: 8px;
    display: block;
}
.wlc-greeting em { font-style: italic; color: #7ecf9a !important; }
.wlc-body { color: #aecebe !important; font-size: 0.865rem; line-height: 1.7; }
.wlc-phone {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 12px;
    padding: 0.38rem 0.9rem;
    background: rgba(76,175,112,0.12);
    border: 1px solid rgba(76,175,112,0.22);
    border-radius: 100px;
    font-size: 0.76rem;
    color: #8ee8a8 !important;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.footer-note {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.67rem;
    color: #b0c8bc;
    margin-top: 0.3rem;
    padding-left: 3px;
    letter-spacing: 0.01em;
}

/* ══════════════════════ SUGGESTION CHIPS ══════════════════════ */
.sug-label {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0.8rem 0 0.55rem;
    display: block;
}

div[data-testid="column"] .stButton > button {
    background: var(--white) !important;
    color: var(--text-mid) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 100px !important;
    font-size: 0.74rem !important;
    padding: 0.42rem 0.85rem !important;
    font-weight: 500 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    width: 100% !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.18s !important;
    transform: none !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: var(--green-dark) !important;
    color: #d8f0e2 !important;
    border-color: var(--green-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(26,58,42,0.2) !important;
}

/* Chip rows: tighten vertical gap between row 1 and row 2 */
div[data-testid="column"] .stButton {
    margin-bottom: 0 !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important;
    row-gap: 0.45rem !important;
    flex-wrap: wrap !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are the AI assistant for ZenShine Pro, a professional cleaning company in Chennai, Tamil Nadu.

== SERVICES & EXACT PRICING ==
- Full House Cleaning: ₹2,000 (up to 2BHK, 3–5 hours)
- Bathroom Cleaning: ₹500 per bathroom (45–60 min each)
- Lawn & Garden Cleaning: ₹800 (up to 500 sq ft, 1–2 hours)
- Floor Cleaning & Polishing: ₹1,000 (1–2 hours)
- Door & Window Cleaning: ₹300 (30–45 min)
- Kitchen Deep Cleaning: ₹1,200 (1.5–2 hours)
- Sofa & Upholstery Cleaning: ₹700 per set (1–1.5 hours)
- Mattress Deep Cleaning: ₹600 per mattress (45–60 min)
- Water Tank Cleaning: ₹1,500 per tank up to 1000L (2–3 hours)
- Pest-Deterrent Sanitization: ₹900 (2–3 hours)
- Post-Construction Cleanup: from ₹3,500 for 1BHK (site visit required)
- Office & Commercial Cleaning: custom quote

== DISCOUNT RULES (apply automatically) ==
- 2 services booked together → 10% OFF total
- 3 or more services → 20% OFF total
- First-time customer → extra ₹100 OFF (stackable)
- Monthly contract → flat 15% OFF all services
- Refer a friend → both get ₹150 OFF next booking
- FREE Door & Window Cleaning (₹300) with any Full House Cleaning booking

== POPULAR PACKAGES ==
- Fresh Home Pack: House + Bath + Floor = ₹3,500 → ₹2,800 after 20% off (save ₹700)
- Kitchen & Hygiene Pack: Kitchen + Bath = ₹1,700 → ₹1,530 after 10% off (save ₹170)
- Premium Full Home: House + Bath + Kitchen + Floor + Sofa + Door + Lawn = ₹7,500 → ₹6,000 after 20% off (save ₹1,500)
- New Home Starter: Post-Construction + Floor + Water Tank = from ₹5,500 (custom)

== CONTACT & OPERATIONS ==
- Phone/WhatsApp: +91 98765 43210
- Email: queries@zenshinepro.in
- Owner: Rahul Sharma (rahul@zenshinepro.in)
- Address: 12 Green Park Street, T Nagar, Chennai 600017
- Hours: Mon–Sat 8AM–8PM, Sun 9AM–5PM
- Same-day service available if booked before 11AM
- Payment: Cash, UPI (GPay/PhonePe/Paytm), bank transfer — collected after service
- No advance payment required

== YOUR BEHAVIOUR RULES ==
1. Always be friendly, warm, and professional.
2. When asked about multiple services, CALCULATE the total with correct discount and show a clear price breakdown.
3. Answer in the same language the customer uses (English, Tamil, or Hindi).
4. Use the FAQ context provided to answer accurately.
5. If a question is outside the FAQ and your knowledge, say so honestly and direct them to call +91 98765 43210.
6. Never make up prices or services not listed above.
7. Keep answers concise but complete. Use bullet points for lists.
8. For greetings, respond warmly and ask how you can help.
"""

# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ══════════════════════════════════════════════════════════════════════════
# GROQ CALL
# ══════════════════════════════════════════════════════════════════════════
def ask_groq(user_query: str, rag_context: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        return (
            "API key not found. "
            "Please add your GROQ_API_KEY to the .env file and restart the app."
        )

    client = Groq(
        api_key=groq_key,
        http_client=httpx.Client(verify=False),
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if rag_context:
        messages.append({
            "role": "system",
            "content": f"== RELEVANT FAQ CONTEXT ==\n{rag_context}"
        })
    for turn in st.session_state.history[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Something went wrong ({type(e).__name__}): {e}"


def get_bot_reply(query: str) -> str:
    rag_context = ""
    rag_error   = None
    try:
        chunks = retrieve(query)
        if chunks:
            rag_context = "\n\n".join(chunks)
    except Exception as e:
        rag_error = str(e)

    if rag_error:
        st.warning(f"Knowledge base retrieval failed — answering from built-in data only.\n\n`{rag_error}`")

    return ask_groq(query, rag_context)


# ══════════════════════════════════════════════════════════════════════════
# SVG ASSETS — no emojis, all SVG icons
# ══════════════════════════════════════════════════════════════════════════

AVATAR_SVG = """<svg viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:38px;height:38px;display:block;">
  <circle cx="19" cy="19" r="19" fill="url(#avBg)"/>
  <circle cx="19" cy="14.5" r="5.5" fill="#2d5a3d"/>
  <path d="M7 36c0-6.627 5.373-10 12-10s12 3.373 12 10" fill="#2d5a3d"/>
  <defs>
    <linearGradient id="avBg" x1="0" y1="0" x2="38" y2="38" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#e8f5ee"/>
      <stop offset="100%" stop-color="#cce8d8"/>
    </linearGradient>
  </defs>
</svg>"""

HEADER_SVG = """<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="14" cy="10" r="5" fill="#7ecf9a"/>
  <path d="M3 27c0-6.075 4.925-9 11-9s11 2.925 11 9" fill="#7ecf9a" fill-opacity="0.8"/>
  <circle cx="23" cy="5" r="2" fill="#a8e8b8"/>
  <path d="M23 2v1M23 7v1M20 5h1M25 5h1" stroke="#a8e8b8" stroke-width="1" stroke-linecap="round"/>
</svg>"""

SB_LOGO_SVG = """<svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M11 1.5 13.2 8H20L14.5 12L16.7 18.5 11 14.5 5.3 18.5 7.5 12 2 8H8.8L11 1.5Z" fill="#4caf70" fill-opacity="0.9"/>
</svg>"""

# Professional SVG icons for sidebar contact & status
ICO_PHONE = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M2.5 1.5C2.5 1.5 1 1.5 1 3.5C1 8.5 4.5 12 9.5 12C11.5 12 11.5 10.5 11.5 10.5V9C11.5 9 11.5 8 10.5 7.5C9.5 7 8.5 8 8.5 8C8.5 8 8 8.5 7 7.5C6 6.5 6.5 6 6.5 6C6.5 6 7.5 5 7 4C6.5 3 5.5 3 5.5 3L4 1.5C4 1.5 3.5 1.5 2.5 1.5Z" stroke="#8aab96" stroke-width="1" stroke-linejoin="round"/>
</svg>"""

ICO_MAIL = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="1" y="3" width="11" height="7.5" rx="1.2" stroke="#8aab96" stroke-width="1"/>
  <path d="M1 4l5.5 3.5L12 4" stroke="#8aab96" stroke-width="1" stroke-linecap="round"/>
</svg>"""

ICO_PIN = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M6.5 1C4.567 1 3 2.567 3 4.5C3 7.5 6.5 12 6.5 12C6.5 12 10 7.5 10 4.5C10 2.567 8.433 1 6.5 1Z" stroke="#8aab96" stroke-width="1"/>
  <circle cx="6.5" cy="4.5" r="1.3" stroke="#8aab96" stroke-width="1"/>
</svg>"""

ICO_CLOCK = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="6.5" cy="6.5" r="5.5" stroke="#8aab96" stroke-width="1"/>
  <path d="M6.5 3.5v3l2.5 1.5" stroke="#8aab96" stroke-width="1" stroke-linecap="round"/>
</svg>"""

ICO_CLOCK_MAIN = """<svg width="12" height="12" viewBox="0 0 12 12" fill="none">
  <circle cx="6" cy="6" r="5.5" stroke="#b0c8bc" stroke-width="1"/>
  <path d="M6 3.5v3l2 1" stroke="#b0c8bc" stroke-width="1" stroke-linecap="round"/>
</svg>"""

ICO_DB = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="6.5" cy="3.5" rx="4.5" ry="1.8" stroke="currentColor" stroke-width="1"/>
  <path d="M2 3.5v2.5c0 1 2 1.8 4.5 1.8s4.5-.8 4.5-1.8V3.5" stroke="currentColor" stroke-width="1"/>
  <path d="M2 6v2.5c0 1 2 1.8 4.5 1.8s4.5-.8 4.5-1.8V6" stroke="currentColor" stroke-width="1"/>
</svg>"""

ICO_CHECK = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="6.5" cy="6.5" r="5.5" stroke="#5fdc88" stroke-width="1"/>
  <path d="M4 6.5l2 2 3-3" stroke="#5fdc88" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

ICO_CROSS = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="6.5" cy="6.5" r="5.5" stroke="#e07070" stroke-width="1"/>
  <path d="M4.5 4.5l4 4M8.5 4.5l-4 4" stroke="#e07070" stroke-width="1.2" stroke-linecap="round"/>
</svg>"""

ICO_UPLOAD = """<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M6.5 1v7M4 3.5l2.5-2.5L9 3.5" stroke="rgba(160,200,175,0.6)" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M2 10h9" stroke="rgba(160,200,175,0.6)" stroke-width="1" stroke-linecap="round"/>
</svg>"""

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-logo-row">
            <div class="sb-logo-box">{SB_LOGO_SVG}</div>
            <div>
                <div class="sb-wordmark">Zen<span>Shine</span> Pro</div>
                <div class="sb-tagline">Chennai's Premium Cleaning</div>
            </div>
        </div>
        <div class="sb-live-pill">
            <span class="sb-live-dot"></span>
            <span class="sb-live-text">AI Assistant &middot; Live</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── System Status (read-only, no input fields) ──
    st.markdown("""<div class="sb-section">
        <span class="sb-section-label">System Status</span>
        <span class="sb-section-line"></span>
    </div>""", unsafe_allow_html=True)

    groq_ok   = bool(os.getenv("GROQ_API_KEY", "").strip())
    qdrant_ok = bool(os.getenv("QDRANT_URL", "").strip() and os.getenv("QDRANT_API_KEY", "").strip())

    groq_icon   = ICO_CHECK if groq_ok   else ICO_CROSS
    qdrant_icon = ICO_CHECK if qdrant_ok else ICO_CROSS
    groq_cls    = "connected"   if groq_ok   else "disconnected"
    qdrant_cls  = "connected"   if qdrant_ok else "disconnected"
    groq_txt    = "Connected"   if groq_ok   else "Not configured"
    qdrant_txt  = "Connected"   if qdrant_ok else "Not configured"

    st.markdown(f"""
    <div class="status-card">
        <div class="status-row">
            <span class="status-label">
                <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                  <rect x="1.5" y="2" width="10" height="7" rx="1" stroke="#8aab96" stroke-width="1"/>
                  <path d="M4 11h5M6.5 9v2" stroke="#8aab96" stroke-width="1" stroke-linecap="round"/>
                </svg>
                Language Model
            </span>
            <span class="status-indicator {groq_cls}">{groq_icon}&nbsp;{groq_txt}</span>
        </div>
        <div class="status-row">
            <span class="status-label">
                {ICO_DB}
                Knowledge Base
            </span>
            <span class="status-indicator {qdrant_cls}">{qdrant_icon}&nbsp;{qdrant_txt}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Knowledge base upload
    st.markdown("""<div class="sb-section">
        <span class="sb-section-label">Knowledge Base</span>
        <span class="sb-section-line"></span>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload FAQ PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.read())
        with st.spinner("Indexing document..."):
            try:
                added = ingest_pdf(path)
                st.success("Already indexed." if added == 0 else f"Indexed {added} sections.")
            except Exception as e:
                st.error(f"Failed: {e}")

    try:
        count = collection_count()
        if count > 0:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;'
                f'color:rgba(120,180,140,0.65);margin-top:0.4rem;padding-left:4px;">'
                f'{ICO_DB.replace("currentColor","rgba(120,180,140,0.65)")}'
                f'&nbsp;{count} sections indexed</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("No document indexed yet.")
    except Exception:
        st.warning("Knowledge base not connected.")

    st.divider()

    # Services & pricing
    st.markdown("""<div class="sb-section">
        <span class="sb-section-label">Services &amp; Pricing</span>
        <span class="sb-section-line"></span>
    </div>""", unsafe_allow_html=True)

    services = [
        ("Full House Cleaning",  "₹2,000"),
        ("Bathroom Cleaning",    "₹500 / bath"),
        ("Lawn & Garden",        "₹800"),
        ("Floor Polishing",      "₹1,000"),
        ("Door & Window",        "₹300"),
        ("Kitchen Deep Clean",   "₹1,200"),
        ("Sofa & Upholstery",    "₹700 / set"),
        ("Mattress Deep Clean",  "₹600 / pc"),
        ("Water Tank",           "₹1,500 / tank"),
        ("Pest Sanitization",    "₹900"),
        ("Post-Construction",    "₹3,500+"),
        ("Office Cleaning",      "Custom"),
    ]
    svc_html = '<div class="svc-list">'
    for name, price in services:
        svc_html += f'<div class="svc-item"><span class="svc-name">{name}</span><span class="svc-price">{price}</span></div>'
    svc_html += '</div>'
    st.markdown(svc_html, unsafe_allow_html=True)

    st.divider()

    # Discounts
    st.markdown("""<div class="sb-section">
        <span class="sb-section-label">Discounts &amp; Offers</span>
        <span class="sb-section-line"></span>
    </div>""", unsafe_allow_html=True)

    discounts = [
        ("2 services booked",  "<b>10% OFF</b> total"),
        ("3+ services",        "<b>20% OFF</b> total"),
        ("First-time visit",   "Extra <b>100 OFF</b>"),
        ("Monthly contract",   "Flat <b>15% OFF</b>"),
        ("Refer a friend",     "Both get <b>150 OFF</b>"),
    ]
    disc_html = '<div class="disc-list">'
    for label, val in discounts:
        disc_html += f'<div class="disc-item"><span class="disc-dot"></span><span class="disc-text">{label} — {val}</span></div>'
    disc_html += '</div>'
    st.markdown(disc_html, unsafe_allow_html=True)

    st.divider()

    # Contact
    st.markdown("""<div class="sb-section">
        <span class="sb-section-label">Contact</span>
        <span class="sb-section-line"></span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="contact-card">
        <div class="contact-row">
            <span class="contact-ico">{ICO_PHONE}</span>
            <span class="contact-detail"><b>+91 98765 43210</b>Call or WhatsApp</span>
        </div>
        <div class="contact-row">
            <span class="contact-ico">{ICO_MAIL}</span>
            <span class="contact-detail"><b>queries@zenshinepro.in</b>Email support</span>
        </div>
        <div class="contact-row">
            <span class="contact-ico">{ICO_PIN}</span>
            <span class="contact-detail"><b>T Nagar, Chennai 600017</b>12 Green Park Street</span>
        </div>
        <div class="contact-row">
            <span class="contact-ico">{ICO_CLOCK}</span>
            <span class="contact-detail"><b>Mon–Sat 8AM–8PM</b>Sun 9AM–5PM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# MAIN — HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
    <div class="header-left">
        <div class="header-icon">{HEADER_SVG}</div>
        <div>
            <div class="main-title">ZenShine <em>Pro</em></div>
            <div class="main-subtitle">AI-powered assistant &middot; Services, pricing &amp; bookings</div>
        </div>
    </div>
    <div class="header-badge">
        <span class="hb-dot"></span>
        <span class="hb-text">Assistant Online</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CHAT WINDOW
# ══════════════════════════════════════════════════════════════════════════
chat_html = '<div class="chat-container" id="chat-box">'

if not st.session_state.history:
    chat_html += f"""
    <div class="bubble-row-bot">
        <div class="bot-avatar">{AVATAR_SVG}</div>
        <div class="welcome-bubble">
            <span class="wlc-greeting">Welcome to ZenShine <em>Pro</em></span>
            <span class="wlc-body">
                I'm your personal assistant &mdash; here to help you with
                <b style="color:#8ee8a8">services</b>, <b style="color:#8ee8a8">pricing</b>,
                <b style="color:#8ee8a8">packages</b>, and <b style="color:#8ee8a8">bookings</b>.<br><br>
                Feel free to ask me anything, and I'll find the best option for you.
            </span>
            <div class="wlc-phone">
                <svg width="11" height="11" viewBox="0 0 13 13" fill="none">
                  <path d="M2.5 1.5C2.5 1.5 1 1.5 1 3.5C1 8.5 4.5 12 9.5 12C11.5 12 11.5 10.5 11.5 10.5V9C11.5 9 11.5 8 10.5 7.5C9.5 7 8.5 8 8.5 8C8.5 8 8 8.5 7 7.5C6 6.5 6.5 6 6.5 6C6.5 6 7.5 5 7 4C6.5 3 5.5 3 5.5 3L4 1.5C4 1.5 3.5 1.5 2.5 1.5Z" stroke="#8ee8a8" stroke-width="1" stroke-linejoin="round"/>
                </svg>
                &nbsp;+91 98765 43210 &nbsp;&middot;&nbsp; Available Mon–Sun
            </div>
        </div>
    </div>
    """
else:
    for msg in st.session_state.history:
        role = msg["role"]
        text = msg["content"].replace("\n", "<br>")
        if role == "user":
            chat_html += f"""
            <div class="bubble-row-user">
                <div class="bubble-user">{text}</div>
            </div>"""
        else:
            chat_html += f"""
            <div class="bubble-row-bot">
                <div class="bot-avatar">{AVATAR_SVG}</div>
                <div class="bubble-bot">{text}</div>
            </div>"""

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

st.markdown("""
<script>
    const box = document.getElementById('chat-box');
    if (box) box.scrollTop = box.scrollHeight;
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# INPUT ROW
# ══════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([9, 1])
with col1:
    user_input = st.text_input(
        label="msg",
        label_visibility="collapsed",
        placeholder="Ask about services, pricing, packages, or how to book…",
        key=f"user_input_{st.session_state.input_key}",
    )
with col2:
    send = st.button("Send")

st.markdown(f"""
<div class="footer-note">
    {ICO_CLOCK_MAIN}
    Responses are AI-generated. For urgent queries, please call us at +91 98765 43210.
</div>
""", unsafe_allow_html=True)

# Handle send
if send and user_input.strip():
    query = user_input.strip()
    st.session_state.history.append({"role": "user", "content": query})
    with st.spinner("Getting your answer…"):
        reply = get_bot_reply(query)
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.input_key += 1
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# SUGGESTION CHIPS — shown only on empty chat, split across 2 rows
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.history:
    st.markdown('<span class="sug-label">Try asking</span>', unsafe_allow_html=True)

    suggestions = [
        "Lawn cleaning cost?",
        "What's included in full house clean?",
        "House + kitchen + bath total?",
        "Any combo discounts?",
        "How do I book a service?",
    ]

    # Row 1 — 3 chips
    row1 = st.columns(3)
    for i in range(3):
        with row1[i]:
            if st.button(suggestions[i], key=f"sug_{i}"):
                st.session_state.history.append({"role": "user", "content": suggestions[i]})
                with st.spinner("Getting your answer…"):
                    reply = get_bot_reply(suggestions[i])
                st.session_state.history.append({"role": "assistant", "content": reply})
                st.session_state.input_key += 1
                st.rerun()

    # Row 2 — 2 chips (padded with empty cols so they don't stretch full width)
    row2 = st.columns([1, 1, 1])
    for i, col_idx in enumerate([0, 1]):
        with row2[col_idx]:
            sug = suggestions[3 + i]
            if st.button(sug, key=f"sug_{3 + i}"):
                st.session_state.history.append({"role": "user", "content": sug})
                with st.spinner("Getting your answer…"):
                    reply = get_bot_reply(sug)
                st.session_state.history.append({"role": "assistant", "content": reply})
                st.session_state.input_key += 1
                st.rerun()