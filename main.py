import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai  # Gemini API

# Replace with your actual Gemini API key
API_KEY = "AIzaSyCucXL59n1dQz--ydkiFQzueDyBopKxW3g"

# Streamlit page configuration
st.set_page_config(page_title="Epsilon Technologies Search", layout="centered", initial_sidebar_state="collapsed")

# Inject custom CSS for gradient background and styling
st.markdown(
    """
    <style>
    /* Gradient background for the whole app */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Gradient text for the main title */
    .title-gradient {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.25rem;
        color: #ddd;
        margin-bottom: 2rem;
    }
    
    /* Styled input box */
    div[data-baseweb="input"] > input {
        border-radius: 12px !important;
        border: 2px solid #feb47b !important;
        padding: 12px !important;
        font-size: 1.1rem !important;
        background-color: #203a43 !important;
        color: #f0f0f0 !important;
    }
    div[data-baseweb="input"] > input:focus {
        border-color: #ff7e5f !important;
        box-shadow: 0 0 8px #ff7e5f !important;
        outline: none !important;
    }
    
    /* Result container styling */
    .result-container {
        background: rgba(255, 126, 95, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin-top: 1.5rem;
        font-size: 1.15rem;
        line-height: 1.5;
        border: 1px solid #feb47b;
        color: #fff;
        box-shadow: 0 4px 15px rgba(255, 126, 95, 0.3);
    }

    /* Error message styling */
    .stError {
        background-color: #ff4c4c;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load FAISS index and dataset mapping
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("company_faiss.index")
    with open("company_mapping.pkl", "rb") as f:
        dataset = pickle.load(f)
    return index, dataset

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini API
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Load resources
index, dataset = load_faiss_index()
model = load_model()

# UI Title with gradient text
st.markdown('<h1 class="title-gradient">Epsilon Technologies</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask any question about our company profile below.</p>', unsafe_allow_html=True)

# User query input
query = st.text_input("Enter your question")

if query:
    # Embed the user query
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    top_k = 2
    distances, indices = index.search(query_embedding, top_k)

    # Extract the most relevant text from dataset
    answer = dataset[indices[0][0]]
    additional_answer = dataset[indices[0][1]]

    # Call Gemini API using query + document context
    prompt = f"""You are an assistant with access to the company profile below.

Company Context:
{answer}

Additional Context:
{additional_answer}

User Question:
{query}

Based on the company information, answer the user's question as clearly and helpfully as possible.
"""

    # Generate response from Gemini
    try:
        gemini_response = gemini_model.generate_content(prompt)
        st.markdown(f'<div class="result-container"><h3>THE RESULT:</h3><p>{gemini_response.text}</p></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error from Gemini API: {e}")
