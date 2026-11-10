import html
import pickle
import time
from datetime import datetime
from pathlib import Path

import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.set_page_config(page_title="Spam Detector", page_icon="💬", layout="wide")

BASE = Path(__file__).parent


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_artifacts():
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)
    with open(BASE / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


ps = PorterStemmer()


def preprocess(text: str) -> str:
    """Exactly the same cleaning used when the model was trained."""
    tokens = nltk.word_tokenize(text.lower())
    stops = set(stopwords.words("english"))
    words = [w for w in tokens if w.isalnum()]
    words = [w for w in words if w not in stops and w not in string.punctuation]
    return " ".join(ps.stem(w) for w in words)


SPAM_THRESHOLD = 0.35  # lower = catches more spam (but more false alarms)


def is_spam(text: str) -> bool:
    model, vectorizer = load_artifacts()
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X)[0][1])
        print(f"[debug] cleaned={cleaned!r} spam_probability={p:.3f}", flush=True)
        return p >= SPAM_THRESHOLD
    pred = model.predict(X)[0]
    return str(pred).strip().lower() in ("1", "spam", "true")


def render(s: str) -> None:
    """Flatten HTML so Streamlit's markdown never treats it as a code block."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


# ---------------------------------------------------------------- CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  background:
    radial-gradient(ellipse 60% 45% at 14% 6%, rgba(52,80,170,.35), transparent 70%),
    radial-gradient(ellipse 45% 40% at 95% 38%, rgba(104,60,200,.35), transparent 70%),
    radial-gradient(ellipse 40% 35% at 100% 100%, rgba(90,60,190,.30), transparent 70%),
    linear-gradient(160deg, #0a0e27 0%, #10143a 55%, #090c22 100%);
  background-attachment: fixed;
}

/* hide Streamlit chrome + every loading / running effect */
header[data-testid="stHeader"], footer, #MainMenu,
[data-testid="stSidebar"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSkeleton"] { display: none !important; }
[data-stale="true"], .stale-element, [data-testid="stale-element"] {
  opacity: 1 !important; transition: none !important; filter: none !important;
}
* { scroll-behavior: auto !important; }

.block-container { max-width: 980px !important; padding: 1.2rem 2rem 2rem 2rem !important; }

/* ---- hero ---- */
.hero { display: flex; justify-content: center; margin: 0; }
.hero svg { width: 430px; max-width: 90%; height: auto; overflow: visible; }
.sub { text-align: center; color: rgba(205,210,250,.75); font-size: 1.25rem; font-weight: 600; margin: 4px 0 22px 0; }

/* ---- chat panel ---- */
.st-key-chat_panel {
  background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
  border: 1.5px solid rgba(165,155,235,.38);
  border-radius: 38px;
  padding: 30px 30px 24px 30px;
  box-shadow: 0 30px 80px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.08);
}

/* scroll area: newest message stays at the bottom, like WhatsApp */
.chat-scroll {
  height: 400px; overflow-y: auto; overflow-x: hidden;
  display: flex; flex-direction: column-reverse;
  padding: 0 8px 0 0;
  scrollbar-width: thin; scrollbar-color: rgba(150,150,230,.45) transparent;
}
.msgs { display: flex; flex-direction: column; gap: 20px; }
.empty { margin: auto; color: rgba(200,205,245,.45); font-size: 1.15rem; text-align: center; }

.bubble-user {
  align-self: flex-end; max-width: 75%;
  background: linear-gradient(135deg, #4b48cc, #3a38aa);
  border: 1.5px solid #6f6cf2;
  border-radius: 28px 28px 4px 28px;
  padding: 16px 28px 10px 28px; color: #fff;
  box-shadow: 0 10px 30px rgba(50,48,160,.35);
  animation: pop .18s ease-out;
}
.bubble-user .t { font-size: 1.3rem; line-height: 1.4; font-weight: 500; word-break: break-word; }
.meta { display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 4px;
        font-size: .9rem; color: rgba(255,255,255,.85); }

.bubble-res {
  align-self: flex-start; display: flex; align-items: center; gap: 16px; position: relative;
  border-radius: 28px 28px 28px 4px; padding: 12px 30px 12px 24px; min-width: 300px;
  animation: pop .18s ease-out;
}
.bubble-res.spam { background: linear-gradient(135deg, rgba(120,48,66,.85), rgba(98,42,60,.85)); border: 1.5px solid #8d4a5d; }
.bubble-res.ham  { background: linear-gradient(135deg, rgba(28,105,68,.85), rgba(24,88,58,.85)); border: 1.5px solid #3f9a68; }
.bubble-res .badge { width: 68px; height: 68px; flex: 0 0 68px; }
.bubble-res .lbl { font-size: 1.9rem; font-weight: 700; color: #fff; line-height: 1.1; padding-bottom: 18px; }
.bubble-res .time { position: absolute; right: 24px; bottom: 10px; font-size: .95rem; color: rgba(255,255,255,.7); }

@keyframes pop { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* ---- input bar ---- */
[data-testid="stForm"] {
  margin-top: 22px;
  background: rgba(24,29,68,.75);
  border: 1.5px solid rgba(150,150,215,.30);
  border-radius: 999px;
  padding: 10px 12px 10px 14px;
}
[data-testid="stForm"]:focus-within { border-color: rgba(130,130,255,.7); }
[data-testid="stForm"] [data-testid="stHorizontalBlock"] { gap: .4rem; align-items: center; }


[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
  background: transparent !important; border: none !important; box-shadow: none !important; outline: none !important;
}
[data-testid="stTextInput"] input {
  background: transparent !important; color: #e3e6ff !important;
  font-size: 1.3rem !important; padding-left: 18px !important;
  border: none !important; box-shadow: none !important; outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(200,205,245,.5) !important; opacity: 1 !important; }
[data-testid="InputInstructions"] { display: none !important; }

[data-testid="stFormSubmitButton"] { display: flex; justify-content: flex-end; }
[data-testid="stFormSubmitButton"] button {
  width: 62px; height: 62px; min-height: 62px; padding: 0; border-radius: 50%; border: none;
  background-color: #5a56ee;
  background-image: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M2.5 20.5L22 12 2.5 3.5l.01 6.6L16 12 2.5 13.9z'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: 46% 50%; background-size: 30px;
  box-shadow: 0 8px 22px rgba(90,86,238,.55);
}
[data-testid="stFormSubmitButton"] button:hover { background-color: #6b67ff; }
[data-testid="stFormSubmitButton"] button p,
[data-testid="stFormSubmitButton"] button div { display: none !important; }

@media (max-width: 700px) {
  .block-container { padding: 1rem !important; }
  .bubble-user .t { font-size: 1.1rem; }
}
</style>
"""

# ---------------------------------------------------------------- SVG pieces
LOGO = """
<div class="hero">
<svg viewBox="0 0 420 250" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="gBlue" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5b8cff"/><stop offset="1" stop-color="#2a38d0"/></linearGradient>
<linearGradient id="gRed" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff6a5f"/><stop offset="1" stop-color="#d4202f"/></linearGradient>
<linearGradient id="gGreen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#34e06f"/><stop offset="1" stop-color="#0e9a40"/></linearGradient>
<linearGradient id="gMail" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#d5dbfb"/></linearGradient>
<filter id="sh" x="-20%" y="-20%" width="150%" height="160%"><feDropShadow dx="0" dy="8" stdDeviation="7" flood-color="#000" flood-opacity=".5"/></filter>
<filter id="ts"><feDropShadow dx="0" dy="3" stdDeviation="2" flood-color="#000" flood-opacity=".35"/></filter>
</defs>
<g filter="url(#sh)">
<path d="M52 176 L38 228 L104 184 Z" fill="#2f45d8"/>
<rect x="14" y="62" width="168" height="126" rx="40" fill="url(#gBlue)" stroke="#86a8ff" stroke-width="3"/>
<rect x="46" y="92" width="104" height="68" rx="13" fill="url(#gMail)"/>
<path d="M52 101 L98 134 L144 101" fill="none" stroke="#3a4bd6" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="112" cy="150" r="6" fill="#3a4bd6"/><circle cx="128" cy="150" r="6" fill="#3a4bd6"/><circle cx="144" cy="150" r="6" fill="#3a4bd6"/>
</g>
<g filter="url(#sh)">
<path d="M186 76 L180 112 L222 80 Z" fill="#d4202f"/>
<rect x="160" y="8" width="212" height="76" rx="32" fill="url(#gRed)" stroke="#ff9a92" stroke-width="2.5"/>
<text x="266" y="64" text-anchor="middle" font-family="Arial Black, Inter, Arial, sans-serif" font-weight="900" font-size="46" fill="#fff" filter="url(#ts)">SPAM</text>
</g>
<g filter="url(#sh)">
<path d="M226 160 L214 196 L258 164 Z" fill="#0e9a40"/>
<rect x="212" y="96" width="196" height="76" rx="32" fill="url(#gGreen)" stroke="#8ff0b0" stroke-width="2.5"/>
<text x="310" y="152" text-anchor="middle" font-family="Arial Black, Inter, Arial, sans-serif" font-weight="900" font-size="46" fill="#fff" filter="url(#ts)">HAM</text>
</g>
</svg>
</div>
<div class="sub">Fraud SMS Detector</div>
"""

BADGE_SPAM = """
<svg class="badge" viewBox="0 0 48 48"><circle cx="24" cy="24" r="24" fill="#e5484d"/>
<path d="M24 10 L36 14.5 V25 C36 32 30.5 36.5 24 39 C17.5 36.5 12 32 12 25 V14.5 Z" fill="none" stroke="#fff" stroke-width="2.6" stroke-linejoin="round"/>
<path d="M24 19 V26.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/><circle cx="24" cy="31" r="1.9" fill="#fff"/></svg>
"""

BADGE_HAM = """
<svg class="badge" viewBox="0 0 48 48"><circle cx="24" cy="24" r="24" fill="#22b866"/>
<path d="M24 10 L36 14.5 V25 C36 32 30.5 36.5 24 39 C17.5 36.5 12 32 12 25 V14.5 Z" fill="none" stroke="#fff" stroke-width="2.6" stroke-linejoin="round"/>
<path d="M18 24.5 L22.5 29 L30.5 20" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>
"""

CHECKS = (
    '<svg width="22" height="16" viewBox="0 0 24 16" fill="none" stroke="#fff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M1 9l4 4L13 3"/><path d="M9 11l2 2L20 3"/></svg>'
)


# ---------------------------------------------------------------- messages
if "messages" not in st.session_state:
    st.session_state.messages = []


def messages_html() -> str:
    msgs = st.session_state.messages
    if not msgs:
        return '<div class="chat-scroll"><div class="empty">Type a message below to check if it is spam or ham.</div></div>'
    out = ['<div class="chat-scroll"><div class="msgs">']
    for m in msgs[-50:]:
        safe = html.escape(m["text"]).replace("$", "&#36;")
        out.append(
            f'<div class="bubble-user"><div class="t">{safe}</div>'
            f'<div class="meta"><span>{m["time"]}</span>{CHECKS}</div></div>'
        )
        if m["spam"] is True:
            out.append(
                f'<div class="bubble-res spam">{BADGE_SPAM}<div class="lbl">Spam</div>'
                f'<div class="time">{m["time"]}</div></div>'
            )
        elif m["spam"] is False:
            out.append(
                f'<div class="bubble-res ham">{BADGE_HAM}<div class="lbl">Ham</div>'
                f'<div class="time">{m["time"]}</div></div>'
            )
    out.append("</div></div>")
    return "".join(out)


# ---------------------------------------------------------------- chat (reruns only this part)
@st.fragment
def chat():
    slot = st.empty()

    with st.form("msg_form", clear_on_submit=True):
        c1, c2 = st.columns([1, 0.1], vertical_alignment="center")
        text = c1.text_input("message", placeholder="Type a message...", label_visibility="collapsed")
        sent = c2.form_submit_button("Send")

    if sent and text.strip():
        msg = {"text": text.strip(), "spam": None, "time": datetime.now().strftime("%I:%M %p")}
        st.session_state.messages.append(msg)
        slot.markdown(messages_html(), unsafe_allow_html=True)  # your message appears first
        try:
            msg["spam"] = is_spam(msg["text"])
        except Exception as e:
            st.session_state.messages.pop()
            st.error(f"Prediction failed: {e}")
        time.sleep(0.4)  # short pause, then the reply pops in

    slot.markdown(messages_html(), unsafe_allow_html=True)


# ---------------------------------------------------------------- page
render(CSS)

try:
    load_artifacts()
except Exception as e:
    st.error(f"Could not load model.pkl / vectorizer.pkl: {e}")
    st.stop()

render(LOGO)

with st.container(key="chat_panel"):
    chat()
