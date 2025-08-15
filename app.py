import streamlit as st
import json
import base64
import asyncio
import aiohttp  # Added for async HTTP requests

# The `st.set_page_config` should be the first Streamlit command
st.set_page_config(
    page_title="MFG ITIS TCS TEAM - Intelligent Data Centre RCA",
    page_icon="🤖",
    layout="wide"
)

# Function to make the API call to the LLM
async def get_root_cause_from_llm(incident_data, initial_alert):
    """
    Calls the Gemini API to get a root cause analysis.
    
    This function uses aiohttp to send an async HTTP POST request.
    """
    
    # Construct the detailed prompt for the LLM
    prompt = (
        "You are an expert data center operations engineer. Your task is to perform a root cause analysis (RCA) "
        "for a technical incident. Analyze the provided logs and the initial alert to identify the "
        "single most likely root cause. Be concise, direct, and professional in your response. "
        "Avoid conjecture and focus on the technical details. "
        "Your output should be the root cause and a brief explanation.\n\n"
        "Initial Alert: {initial_alert}\n\n"
        "Logs/Data:\n{incident_data}\n\n"
        "Root Cause Analysis:"
    ).format(initial_alert=initial_alert, incident_data=incident_data)

    chatHistory = []
    chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] })
    payload = {
        "contents": chatHistory
    }

    # IMPORTANT: Replace with your actual Gemini API key
    apiKey = "AIzaSyA2W2u4HUZFll-UTlqCrRngAhVIphsrrns"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(apiUrl, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    st.error(f"API returned status code {resp.status}")
                    return None
                response_data = await resp.json()
        
        if response_data.get('candidates'):
            return response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            st.error("No valid response from the API. Please check the logs and try again.")
            st.json(response_data)
            return None
    except Exception as e:
        st.error(f"An error occurred during the API call: {e}")
        return None

# --- Streamlit UI ---
st.title("🤖 MFG ITIS TCS TEAM - Intelligent Data Centre Incident RCA")
st.markdown("---")

st.markdown("""
This application uses a large language model to help you perform a preliminary root cause analysis (RCA) on data centre incidents.
Simply provide the initial alert and relevant log data, and the model will provide a likely root cause.

**Disclaimer:** This is an AI-powered tool. The analysis provided should be considered a starting point and must be verified by a human expert before any action is taken.
""")

# Input fields
st.markdown("### Incident Details")
initial_alert = st.text_input(
    "Initial Alert Message",
    placeholder="e.g., 'High CPU utilization on DB server-01'"
)
incident_data = st.text_area(
    "Log Data & Monitoring Alerts",
    height=300,
    placeholder="Paste all relevant logs, alerts, and metrics here."
)

# Analysis button
if st.button("Perform RCA", type="primary"):
    if not initial_alert or not incident_data:
        st.warning("Please provide both the initial alert and incident data.")
    else:
        st.markdown("---")
        st.subheader("Analysis Result")
        with st.spinner("Analyzing data and generating root cause..."):
            # Safely create an event loop in Python 3.13+
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(get_root_cause_from_llm(incident_data, initial_alert))
            
            if result:
                st.info(result)
            else:
                st.error("Failed to get an analysis result. Please check the provided data and try again.")
