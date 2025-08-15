import streamlit as st
import os
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.google_search import GoogleSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document

# Set up Streamlit page configuration
st.set_page_config(
    page_title="MFG ITIS TEAM - Intelligent Data Centre RCA",
    page_icon="🤖",
    layout="wide",
)

# Load environment variables (API keys)
# Ensure you have a .env file with GOOGLE_API_KEY and GOOGLE_SEARCH_API_KEY
# You can get these from the Google Cloud Console.
# You will also need to enable the Custom Search API.
from dotenv import load_dotenv
load_dotenv()

# --- Gemini API Key and Search API Key Setup ---
# The environment variables are expected to be set for the app to work.
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
if not os.getenv("GOOGLE_SEARCH_API_KEY") or not os.getenv("GOOGLE_SEARCH_API_ID"):
    st.warning("Google Search API keys are not set. The agent will not be able to perform web searches.")

# --- Simulated Data Centre Knowledge Base ---
# In a real-world scenario, this would be a large collection of real logs,
# incident reports, and system documentation.
st.markdown("## 🧠 Simulated Data Base")
with st.expander("Expand to see the simulated log data"):
    data_centre_logs = [
        "Network-01: HIGH latency detected on switch port 3. Packet drops > 50%. BGP peer 10.10.1.5 is flapping.",
        "Server-03: CPU utilization 98% for process `db_backup`. Memory usage 95%. I/O wait is high.",
        "Network-01: Resolved issue with BGP peer. The upstream provider had a configuration error. Connection restored.",
        "Storage-05: Disk full on `/var/log`. No new logs can be written. `df -h` shows 100% usage.",
        "Server-12: The `nginx` service is not responding. A dependency service `auth-service` failed to start.",
        "Database-01: High number of deadlocks in transactions. Query `SELECT * FROM large_table` is running without an index.",
        "Server-03: `db_backup` process completed. CPU and memory usage returned to normal levels. Logs show normal operations.",
        "Network-02: Power supply failure on top-of-rack switch. Traffic re-routed successfully via redundant path.",
        "Server-12: `auth-service` was restarted manually. `nginx` is now responding as expected. No further issues detected.",
    ]
    st.json(data_centre_logs)

# --- RAG System Setup ---
# This function sets up the Retrieval-Augmented Generation system.
@st.cache_resource
def setup_rag_system():
    st.info("Initializing RAG system (Vector Store)...")
    try:
        # Create a text splitter to chunk our data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=log) for log in data_centre_logs]
        texts = text_splitter.split_documents(docs)

        # Create embeddings and a FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(texts, embeddings)
        st.success("RAG system initialized successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

vector_store = setup_rag_system()

# --- Agentic AI Setup ---
# This function sets up the LLM agent with tools.
@st.cache_resource
def setup_agent():
    st.info("Initializing Agentic AI...")
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

        # Set up the Google Search tool
        google_search_tool = GoogleSearchAPIWrapper(k=5)
        tools = [google_search_tool]

        # Define the prompt template for the agent
        # This prompt is critical for guiding the agent's behavior.
        template = """
        You are a highly skilled Data Centre Root Cause Analysis (RCA) expert. Your task is to analyze incident reports and logs to accurately identify the root cause, propose solutions, and explain your reasoning clearly and concisely.

        You have access to the following tools:
        {tools}

        The incident description is: '{input}'
        
        The internal data center logs and information are provided below. This is your primary source of truth.
        ----------------
        {context}
        ----------------

        First, analyze the provided logs and the incident description to find the root cause.
        If the internal logs do not provide a clear answer, or if you need more general information to understand the problem, you may use the Google Search tool.
        
        Your response must follow this structure exactly:
        
        **Incident Summary:**
        [A brief summary of the incident based on the user's input.]
        
        **Root Cause Analysis:**
        [Detailed analysis of the root cause. Reference specific logs from the provided context. If you used Google Search, mention what you learned from it. Explain your logical reasoning.]
        
        **Proposed Solution:**
        [A clear, actionable solution or a set of steps to resolve the issue. Be specific.]
        
        **Preventative Measures:**
        [Suggestions for what could be done to prevent this type of incident from happening again.]
        
        **Confidence Score:**
        [Provide a confidence score from 1-100% on your analysis, explaining why you chose this score.]

        Begin!
        """

        prompt = PromptTemplate.from_template(template)

        # Create the agent
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        st.success("Agentic AI initialized successfully!")
        return agent_executor
    except Exception as e:
        st.error(f"Failed to initialize Agentic AI: {e}")
        return None

agent_executor = setup_agent()

# --- Streamlit UI ---
st.title("🤖 MFG ITIS TEAM - Intelligent Data Centre Incident RCA")
st.markdown(
    """
    This application simulates an AI-powered tool for root cause analysis (RCA) of data centre incidents.
    It uses a Retrieval-Augmented Generation (RAG) system to find relevant logs and an Agentic AI to
    synthesize a detailed report.
    """
)

# User input for the incident
incident_description = st.text_area(
    "**Describe the incident:**",
    value="The website is down. We're seeing slow performance on the home page and authentication is failing.",
    height=150
)

# Button to trigger the analysis
if st.button("Analyze Incident", type="primary", use_container_width=True):
    if not incident_description:
        st.warning("Please provide a description of the incident.")
    else:
        if not vector_store or not agent_executor:
            st.error("Application components failed to initialize. Please check API keys and logs.")
        else:
            with st.spinner("Analyzing incident... This may take a moment."):
                try:
                    # RAG Step: Retrieve relevant logs based on the incident description
                    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                    retrieved_docs = retriever.invoke(incident_description)
                    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
                    
                    st.divider()
                    st.info("### 📝 Relevant Logs and Context from RAG")
                    st.text_area(
                        "RAG System Retrieved the following relevant documents:",
                        context_text,
                        height=200
                    )
                    
                    # Agent Step: Use the LLM to perform the analysis
                    agent_output = agent_executor.invoke({"input": incident_description, "context": context_text})
                    
                    st.divider()
                    st.info("### 🤖 Agentic AI Root Cause Analysis")
                    st.markdown(agent_output["output"])
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
