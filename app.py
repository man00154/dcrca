import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
import traceback

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Streamlit UI ---
st.set_page_config(
    page_title="TCS MFG ITIS - Intelligent Data Centre RCA",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 TCS MFG ITIS - Intelligent Data Centre Incident RCA")
st.markdown("AI-powered tool to generate Root Cause Analysis (RCA) and Solutions.")

incident_description = st.text_area(
    "**Describe the incident:**",
    value="Network-03: Users reporting intermittent connectivity; BGP sessions flapping; high I/O latency on servers.",
    height=150
)

def initialize_llm():
    """Initialize LLM: Try Gemini first, fallback to OpenAI if quota exceeded."""
    llm = None
    try:
        if GOOGLE_API_KEY:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-latest",
                    temperature=0.2,
                    google_api_key=GOOGLE_API_KEY
                )
                llm.invoke("Hello")  # test call
                st.info("Using Gemini LLM.")
            except Exception as gemini_error:
                st.warning(f"⚠ Gemini API unavailable or quota exceeded. Switching to OpenAI.\n{gemini_error}")
                if not OPENAI_API_KEY:
                    st.error("No fallback OpenAI API key found — cannot initialize LLM.")
                    return None
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    openai_api_key=OPENAI_API_KEY
                )
                st.info("Using fallback OpenAI LLM.")
        else:
            if not OPENAI_API_KEY:
                st.error("No API key found — cannot initialize any LLM.")
                return None
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY
            )
            st.info("Using fallback OpenAI LLM.")
    except Exception as e:
        st.error(f"Failed to initialize LLM:\n{traceback.format_exc()}")
        return None
    return llm

llm = initialize_llm()

if st.button("Generate RCA & Solution"):
    if not incident_description.strip():
        st.warning("Please provide a description of the incident.")
    elif not llm:
        st.error("LLM not initialized. Cannot proceed.")
    else:
        st.info("Analyzing incident...")
        try:
            prompt = f"""
You are a skilled Data Centre Root Cause Analysis (RCA) expert.
Incident description: {incident_description}

Provide the RCA and recommended solution in this format:

**Root Cause:** <Explain the technical cause>
**Solution:** <Step-by-step remediation actions>
"""
            response = llm.invoke(prompt)
            st.markdown("### 🤖 RCA & Solution")
            st.markdown(response)
        except Exception as e:
            st.error(f"Error generating RCA:\n{traceback.format_exc()}")
