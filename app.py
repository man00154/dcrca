import streamlit as st
import os
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.google_search.tool import GoogleSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.tools import Tool
from dotenv import load_dotenv

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MFG ITIS TEAM - Intelligent Data Centre RCA",
    page_icon="🤖",
    layout="wide",
)

# --- Load environment variables ---
load_dotenv()

# ✅ Check all required Google Search variables
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
if not os.getenv("GOOGLE_CSE_ID"):
    st.error("Please set the GOOGLE_CSE_ID environment variable.")
    st.stop()

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

        # ✅ Google Search API Wrapper
        google_search_tool = GoogleSearchAPIWrapper(
            k=5,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID")
        )

        # ✅ Wrap search tool into LangChain Tool
        tools = [
            Tool(
                name="google_search",
                func=google_search_tool.run,
                description="Use this tool to search the web for the latest or external information."
            )
        ]

        template = """
        You are a highly skilled Data Centre Root Cause Analysis (RCA) expert...
        {tools}

        The incident description is: '{input}'
        
        The internal data center logs and information are provided below:
        ----------------
        {context}
        ----------------
        ...
        """
        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        st.success("Agentic AI initialized successfully!")
        return agent_executor
    except Exception as e:
        st.error(f"Failed to initialize Agentic AI: {e}")
        return None

agent_executor = setup_agent()

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

                    agent_output = agent_executor.invoke({"input": incident_description, "context": context_text})

                    st.divider()
                    st.info("### 🤖 Agentic AI Root Cause Analysis")
                    st.markdown(agent_output["output"])

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
