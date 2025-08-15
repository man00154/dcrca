import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import hashlib
import pickle

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MFG ITIS TEAM - Intelligent Data Centre RCA",
    page_icon="🤖",
    layout="wide",
)

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Cache Setup ---
CACHE_FILE = "gemini_cache.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        GEMINI_CACHE = pickle.load(f)
else:
    GEMINI_CACHE = {}

if "session_cache" not in st.session_state:
    st.session_state.session_cache = {}

def cache_response(key, response=None):
    global GEMINI_CACHE
    if response is None:
        if key in st.session_state.session_cache:
            return st.session_state.session_cache[key]
        return GEMINI_CACHE.get(key)
    else:
        st.session_state.session_cache[key] = response
        GEMINI_CACHE[key] = response
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(GEMINI_CACHE, f)

# --- Cache Management UI ---
st.sidebar.markdown("## ⚙️ Cache Management")
if st.sidebar.button("Clear Session Cache"):
    st.session_state.session_cache = {}
    st.sidebar.success("✅ Session cache cleared.")
if st.sidebar.button("Clear Disk Cache"):
    GEMINI_CACHE.clear()
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    st.session_state.session_cache = {}
    st.sidebar.success("✅ Disk cache cleared.")

# --- Simulated Logs ---
st.markdown("## 🧠 Simulated Data Base")
with st.expander("Expand to see the simulated log data"):
    data_centre_logs = [
        "Network-01: HIGH latency detected on switch port 3. Packet drops > 50%. BGP peer 10.10.1.5 is flapping.",
        "Server-03: CPU utilization 98% for process `db_backup`. Memory usage 95%. I/O wait is high.",
        "Network-01: Resolved issue with BGP peer. The upstream provider had a configuration error. Connection restored.",
        "Storage-05: Disk full on `/var/log`. No new logs can be written. `df -h` shows 100% usage.",
        "Server-12: `nginx` service is not responding. A dependency service `auth-service` failed to start.",
        "Database-01: High number of deadlocks in transactions. Query `SELECT * FROM large_table` is running without an index.",
        "Server-03: `db_backup` process completed. CPU and memory usage returned to normal levels. Logs show normal operations.",
        "Network-02: Power supply failure on top-of-rack switch. Traffic re-routed successfully via redundant path.",
        "Server-12: `auth-service` was restarted manually. `nginx` is now responding as expected. No further issues detected.",
    ]
    st.json(data_centre_logs)

# --- LLM Setup (Gemini / Google Generative AI) ---
@st.cache_resource
def setup_llm():
    if not GOOGLE_API_KEY:
        st.error("Please set GOOGLE_API_KEY in .env")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
        llm.invoke("Hello")  # Test LLM
        st.success("Gemini LLM initialized successfully!")
        return llm
    except Exception as e:
        st.error(f"Gemini LLM initialization failed: {e}")
        return None

llm = setup_llm()

# --- UI ---
st.title("🤖 MFG ITIS TEAM - Intelligent Data Centre Incident RCA")
st.markdown("AI-powered RCA tool for data centre incidents.")

incident_description = st.text_area(
    "**Describe the incident:**",
    value="The website is down. We're seeing slow performance on the home page and authentication is failing.",
    height=150
)

if st.button("Analyze Incident", type="primary", use_container_width=True):
    if not incident_description:
        st.warning("Please provide a description of the incident.")
    elif not llm:
        st.error("LLM failed to initialize.")
    else:
        with st.spinner("Analyzing incident... This may take a moment."):
            key = hashlib.sha256(incident_description.encode("utf-8")).hexdigest()
            cached_output = cache_response(key)
            if cached_output:
                st.info("Using cached RCA output (API quota saved).")
                st.divider()
                st.info("### 🤖 RCA Output")
                st.markdown(cached_output)
            else:
                try:
                    # --- Create Agentic RCA Prompt ---
                    context_text = "\n".join(data_centre_logs)
                    prompt = f"""
You are a highly skilled Data Centre Root Cause Analysis (RCA) expert.

Incident description:
{incident_description}

Internal logs & context:
{context_text}

Provide the RCA in this format:

**Root Cause:** <Explain the technical cause here>
**Solution:** <Step-by-step remediation actions here>
"""
                    rca_output = llm.invoke(prompt)
                    st.divider()
                    st.info("### 🤖 RCA Output")
                    st.markdown(rca_output)
                    cache_response(key, rca_output)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
