import os
import streamlit as st
import faiss, pickle
import numpy as np
import google.generativeai as genai
import time
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun


st.set_page_config(
    page_title="Epsilon Technologies", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://epsilon-tech.com/help',
        'Report a bug': 'https://epsilon-tech.com/bug',
        'About': "# Epsilon Technologies\nAI-Powered Company Assistant"
    }
)

# Enhanced CSS with modern styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%) !important;
            color: #f5d76e !important;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        .main-header {
            background: linear-gradient(90deg, #f5d76e 0%, #ffd700 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        }

        .subtitle {
            text-align: center;
            color: #cccccc;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        .stats-container {
            display: flex;
            justify-content: space-around;
            margin: 2rem 0;
        }

        .stat-card {
            background: rgba(245, 215, 110, 0.1);
            border: 1px solid rgba(245, 215, 110, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            min-width: 150px;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(245, 215, 110, 0.2);
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #f5d76e;
        }

        .stat-label {
            font-size: 0.9rem;
            color: #cccccc;
            margin-top: 0.5rem;
        }

        h1, h2, h3, h4, h5, h6, label, .stMarkdown {
            color: #f5d76e !important;
            text-shadow: 0 0 6px rgba(255, 215, 0, 0.3);
        }

        .stTextInput > div > div > input {
            background: rgba(26, 26, 26, 0.8) !important;
            color: #f5d76e !important;
            border: 2px solid rgba(245, 215, 110, 0.3) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(10px) !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #f5d76e !important;
            box-shadow: 0 0 20px rgba(245, 215, 110, 0.3) !important;
        }

        .stTextInput label {
            color: #f5d76e !important;
            font-weight: 500 !important;
            margin-bottom: 0.5rem !important;
        }

        .response-box {
            background: linear-gradient(135deg, rgba(28, 28, 28, 0.9) 0%, rgba(34, 34, 34, 0.9) 100%);
            border: 1px solid rgba(245, 215, 110, 0.3);
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            color: #f5d76e;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }

        .response-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #f5d76e, transparent);
        }

        .confidence-indicator {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-left: 10px;
        }

        .confidence-high { background: rgba(76, 175, 80, 0.2); color: #4caf50; }
        .confidence-medium { background: rgba(255, 152, 0, 0.2); color: #ff9800; }
        .confidence-low { background: rgba(244, 67, 54, 0.2); color: #f44336; }

        .stAlert {
            background: rgba(26, 26, 26, 0.9) !important;
            color: #f5d76e !important;
            border-left: 4px solid #f5d76e !important;
            border-radius: 8px !important;
            backdrop-filter: blur(10px) !important;
        }

        .stExpanderHeader {
            color: #f5d76e !important;
            background: rgba(26, 26, 26, 0.5) !important;
            border-radius: 8px !important;
        }

        .stExpanderContent {
            background: rgba(20, 20, 20, 0.9) !important;
            border-radius: 0 0 8px 8px !important;
            backdrop-filter: blur(10px) !important;
        }

        .sidebar .stSelectbox > div > div {
            background: rgba(26, 26, 26, 0.8) !important;
            border: 1px solid rgba(245, 215, 110, 0.3) !important;
            border-radius: 8px !important;
        }

        .typing-indicator {
            display: inline-block;
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .query-examples {
            background: rgba(245, 215, 110, 0.05);
            border: 1px solid rgba(245, 215, 110, 0.2);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }

        .example-query {
            background: rgba(245, 215, 110, 0.1);
            border: 1px solid rgba(245, 215, 110, 0.2);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            text-decoration: none;
            color: #f5d76e;
        }

        .example-query:hover {
            background: rgba(245, 215, 110, 0.2);
            transform: translateX(5px);
        }

        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(245, 215, 110, 0.3);
            border-radius: 50%;
            border-top-color: #f5d76e;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# === Original Logic (Unchanged) ===
API_KEY = "AIzaSyCucXL59n1dQz--ydkiFQzueDyBopKxW3g"
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

@st.cache_resource
def load_search():
    idx = faiss.read_index("company_faiss.index")
    with open("company_mapping.pkl", "rb") as f:
        data = pickle.load(f)
    return idx, data

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index, dataset = load_search()
embedder = load_embedding_model()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, code_execution=True)

search_tool = DuckDuckGoSearchRun()
tools_ddg = [Tool(name="web-search", func=search_tool.run, description="Search the web for information.")]

react_prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to tools:
{tools}

Use the following format:

Thought: Think about the problem step-by-step.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation)
Final Answer: the final answer to the user's question

Begin!

Question: {input}
{agent_scratchpad}"""
)

mem_rag = ConversationBufferMemory(memory_key="history_rag", return_messages=True)
mem_ddg = ConversationBufferMemory(memory_key="history_ddg", return_messages=True)

agent_rag = create_react_agent(llm=llm, tools=[], prompt=react_prompt)
agent_ddg = create_react_agent(llm=llm, tools=tools_ddg, prompt=react_prompt)

exec_rag = AgentExecutor(agent=agent_rag, tools=[], memory=mem_rag, handle_parsing_errors=True, verbose=False)
exec_ddg = AgentExecutor(agent=agent_ddg, tools=tools_ddg, memory=mem_ddg, handle_parsing_errors=True, verbose=False)

# === Enhanced UI ===

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Model temperature slider
    temperature = st.slider("🌡️ Response Temperature", 0.0, 1.0, 0.5, 0.1, 
                          help="Higher values make responses more creative, lower values more focused")
    
    # RAG confidence threshold
    rag_threshold = st.slider("🎯 RAG Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
                             help="Minimum confidence score to use internal data")
    
    # Query mode selection
    query_mode = st.selectbox("🔍 Query Mode", 
                            ["Auto (Smart Routing)", "Force RAG", "Force Web Search"],
                            help="Choose how to handle queries")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Session Stats")
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'rag_count' not in st.session_state:
        st.session_state.rag_count = 0
    if 'web_count' not in st.session_state:
        st.session_state.web_count = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        st.metric("RAG Used", st.session_state.rag_count)
    
    st.metric("Web Searches", st.session_state.web_count)
    
    if st.button("🔄 Reset Stats"):
        st.session_state.query_count = 0
        st.session_state.rag_count = 0
        st.session_state.web_count = 0
        st.rerun()

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">Epsilon Technologies</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Company Assistant with RAG & Web Search</p>', unsafe_allow_html=True)

# Stats cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">🤖</div>
        <div class="stat-label">Gemini 2.0</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">⚡</div>
        <div class="stat-label">Fast RAG</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">🌐</div>
        <div class="stat-label">Web Search</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">🧠</div>
        <div class="stat-label">Smart Routing</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Query examples
with st.expander("💡 Example Queries", expanded=False):
    st.markdown("""
    <div class="query-examples">
        <h4>Try these example queries:</h4>
        <div class="example-query" onclick="document.querySelector('input').value='What is Epsilon Technologies?'">
            📋 What is Epsilon Technologies?
        </div>
        <div class="example-query" onclick="document.querySelector('input').value='Tell me about our company culture'">
            🏢 Tell me about our company culture
        </div>
        <div class="example-query" onclick="document.querySelector('input').value='What are the latest AI trends?'">
            🔍 What are the latest AI trends?
        </div>
        <div class="example-query" onclick="document.querySelector('input').value='How does our RAG system work?'">
            ⚙️ How does our RAG system work?
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main query input
query = st.text_input("🔍 Ask about our company or anything else:", 
                     placeholder="Type your question here...", 
                     help="Ask about company information or general topics")

# Process query
if query:
    st.session_state.query_count += 1
    
    # Update LLM temperature
    llm.temperature = temperature
    
    # Show processing indicator
    with st.spinner("🔄 Processing your query..."):
        # Original logic with confidence threshold from sidebar
        emb = embedder.embed_query(query)
        q_emb = np.array(emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, k=2)
        context, extra = dataset[I[0][0]], dataset[I[0][1]]
        score = float(D[0][0])
        
        # Confidence indicator
        if score >= 0.7:
            confidence_class = "confidence-high"
            confidence_text = "High"
        elif score >= 0.4:
            confidence_class = "confidence-medium"
            confidence_text = "Medium"
        else:
            confidence_class = "confidence-low"
            confidence_text = "Low"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <span>🎯 RAG Confidence Score: <strong>{score:.3f}</strong></span>
            <span class="confidence-indicator {confidence_class}">{confidence_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Determine routing based on mode and confidence
        use_rag = False
        if query_mode == "Force RAG":
            use_rag = True
        elif query_mode == "Force Web Search":
            use_rag = False
        else:  # Auto mode
            use_rag = score >= rag_threshold
        
        if use_rag:
            st.session_state.rag_count += 1
            prompt = f"Context:\n{context}\n\nExtra:\n{extra}\n\nUser: {query}"
            agent_exec = exec_rag
            try:
                result = agent_exec.invoke({"input": prompt})
                response = result["output"]
                st.success("✅ Answered using internal company data (RAG)")
                source_info = "📚 Internal Knowledge Base"
            except Exception as e:
                st.error(f"❌ RAG agent failed: {e}")
                response = "An error occurred during RAG processing."
                source_info = "❌ Error"
        else:
            st.session_state.web_count += 1
            agent_exec = exec_ddg
            try:
                result = agent_exec.invoke({"input": query})
                response = result["output"]
                st.warning("⚠️ Answer generated using web search (DuckDuckGo)")
                source_info = "🌐 Web Search"
            except Exception as e:
                st.error(f"❌ Web agent failed: {e}")
                response = "An error occurred during web search."
                source_info = "❌ Error"
        
        # Enhanced response display
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class='response-box'>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <strong>🤖 Agent Response</strong>
                <div style="font-size: 0.9rem; color: #cccccc;">
                    {source_info} | {timestamp}
                </div>
            </div>
            <div style="line-height: 1.6; font-size: 1rem;">
                {response}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Conversation history with improved formatting
        with st.expander("💬 Conversation History", expanded=False):
            messages = agent_exec.memory.chat_memory.messages
            if messages:
                for i, msg in enumerate(messages):
                    role = "🧑‍💬 You" if msg.type == "human" else "🤖 Assistant"
                    msg_time = datetime.now().strftime("%H:%M")
                    
                    st.markdown(f"""
                    <div style="margin: 10px 0; padding: 10px; 
                                background: {'rgba(245, 215, 110, 0.1)' if msg.type == 'human' else 'rgba(255, 255, 255, 0.05)'};
                                border-radius: 8px; border-left: 3px solid {'#f5d76e' if msg.type == 'human' else '#666'};">
                        <div style="font-size: 0.9rem; color: #cccccc; margin-bottom: 5px;">
                            {role} • {msg_time}
                        </div>
                        <div>{msg.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No conversation history yet. Start by asking a question!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Powered by Gemini 2.0 Flash • Enhanced RAG System • Real-time Web Search
    </div>
    """, unsafe_allow_html=True)