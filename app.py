import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.google_search.tool import GoogleSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.tools import Tool

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MFG ITIS TEAM - Intelligent Data Centre RCA",
    page_icon="🤖",
    layout="wide",
)

# --- Load environment variables ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # for fallback

if not GOOGLE_API_KEY:
    st.warning("⚠ GOOGLE_API_KEY not found — will use fallback model if Gemini is unavailable.")
if not GOOGLE_CSE_ID:
    st.warning("⚠ GOOGLE_CSE_ID not found — Google Search tool may not work.")

# --- Simulated Logs ---
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

# --- RAG Setup ---
@st.cache_resource
def setup_rag_system():
    st.info("Initializing RAG system (Vector Store)...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=log) for log in data_centre_logs]
        texts = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_documents(texts, embeddings)
        st.success("RAG system initialized successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

vector_store = setup_rag_system()

# --- Agent Setup ---
@st.cache_resource
def setup_agent():
    st.info("Initializing Agentic AI...")

    # Step 1: Try Gemini
    try:
        if GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=GOOGLE_API_KEY)
        else:
            raise ValueError("No GOOGLE_API_KEY found")

    # Step 2: Fallback to OpenAI
    except Exception as e:
        st.warning(f"⚠ Gemini API not available or quota exceeded — switching to OpenAI model. ({e})")
        if not OPENAI_API_KEY:
            st.error("No fallback API key found — cannot initialize LLM.")
            return None, None, None
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)

    try:
        # Google Search Tool (optional)
        google_search_tool = None
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            google_search = GoogleSearchAPIWrapper(
                k=5,
                google_api_key=GOOGLE_API_KEY,
                google_cse_id=GOOGLE_CSE_ID
            )
            google_search_tool = Tool(
                name="Google Search",
                description="Useful for searching the internet for real-time information.",
                func=google_search.run
            )

        tools_list = [google_search_tool] if google_search_tool else []
        tool_names_list = ", ".join([tool.name for tool in tools_list]) if tools_list else "None"

        template = """
You are a highly skilled Data Centre Root Cause Analysis (RCA) expert.
You can use the following tools:
{tools}

Tool names available: {tool_names}

Incident description:
{input}

Internal logs & context:
{context}

Reasoning & intermediate steps:
{agent_scratchpad}

Provide the final RCA in this format:

**Root Cause:** <Explain the technical cause here>
**Solution:** <Step-by-step remediation actions here>
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "context", "agent_scratchpad", "tools", "tool_names"]
        )

        agent = create_react_agent(llm, tools_list, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools_list,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=12,
            max_execution_time=90
        )

        st.success("Agentic AI initialized successfully!")
        return agent_executor, tools_list, tool_names_list
    except Exception as e:
        st.error(f"Failed to initialize Agentic AI: {e}")
        return None, None, None

agent_executor, tools, tool_names = setup_agent()

# --- UI ---
st.title("🤖 MFG ITIS TEAM - Intelligent Data Centre Incident RCA")
st.markdown(
    """
    This application simulates an AI-powered tool for root cause analysis (RCA) of data centre incidents.
    """
)

incident_description = st.text_area(
    "**Describe the incident:**",
    value="The website is down. We're seeing slow performance on the home page and authentication is failing.",
    height=150
)

if st.button("Analyze Incident", type="primary", use_container_width=True):
    if not incident_description:
        st.warning("Please provide a description of the incident.")
    else:
        if not vector_store or not agent_executor:
            st.error("Application components failed to initialize.")
        else:
            with st.spinner("Analyzing incident... This may take a moment."):
                try:
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

                    agent_output = agent_executor.invoke({
                        "input": incident_description,
                        "context": context_text,
                        "tools": tools,
                        "tool_names": tool_names
                    })

                    st.divider()
                    st.info("### 🤖 Agentic AI Root Cause Analysis")
                    if isinstance(agent_output, dict) and "output" in agent_output:
                        st.markdown(agent_output["output"])
                    else:
                        st.markdown(str(agent_output))

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
