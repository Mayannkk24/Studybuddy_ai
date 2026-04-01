import streamlit as st
import os
import base64
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv

# Modern, Stable LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- 1. CONFIG & ELITE UI STYLING ---
load_dotenv()
st.set_page_config(page_title="StudyBuddy AI Pro", layout="wide", page_icon="🎓")

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "initial_analysis" not in st.session_state:
    st.session_state.initial_analysis = None

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    .stApp { background-color: #0B0F1A; font-family: 'Plus Jakarta Sans', sans-serif; color: #F8FAFC; }
    header {visibility: hidden;} footer {visibility: hidden;} #MainMenu {visibility: hidden;}
    [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1F2937; }
    .sidebar-brand { padding: 1.5rem; text-align: center; background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(129, 140, 248, 0.1)); border-radius: 16px; margin-bottom: 2rem; border: 1px solid rgba(56, 189, 248, 0.2); }
    .brand-text { font-size: 1.5rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-section { text-align: center; padding: 60px 20px 30px 20px; background: radial-gradient(circle at center, rgba(37, 99, 235, 0.1) 0%, transparent 70%); }
    .hero-title { font-size: 4rem !important; font-weight: 800; margin-bottom: 1rem; letter-spacing: -2px; }
    .feature-card { background: #1E293B; padding: 2rem; border-radius: 24px; border: 1px solid #334155; transition: 0.3s; height: 100%; }
    .stButton>button { width: 100%; border-radius: 14px; height: 3.8rem; background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); color: white !important; font-weight: 700; border: none; }
    .content-card { background: rgba(30, 41, 59, 0.7); padding: 2.5rem; border-radius: 24px; border: 1px solid #334155; backdrop-filter: blur(12px); margin-top: 20px; color: #E2E8F0; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background-color: #1E293B; border-radius: 12px; padding: 10px 20px; color: #94A3B8; }
    .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. API SETUP ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🔑 API Key Missing! Check your .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 3. HELPERS ---
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def get_pdf_text(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        content = page.extract_text()
        if content: text += content
    return text

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="brand-text">STUDYBUDDY AI</div></div>', unsafe_allow_html=True)
    doc_type = st.radio("⚡ Operation Mode", ["Notes Generation", "Exam Prediction"])
    mood = st.selectbox("🎭 AI Teaching Style", ["Academic", "Explain Like I'm 5", "Interview Prep"])
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload Materials", type=["pdf", "png", "jpg", "jpeg"])
    submit = st.button("🚀 Analyze & Train Agent")
    if st.button("🗑️ Reset All"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.session_state.initial_analysis = None
        st.rerun()

# --- 5. HERO INTERFACE ---
if not submit and not st.session_state.vector_db:
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Academic <span style="color:#38BDF8">Intelligence.</span></h1>
            <p style="color:#94A3B8; font-size:1.2rem; max-width:700px; margin:auto;">
                The modern way to study. Upload your materials to build a persistent knowledge base and generate elite revision kits.
            </p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown('<div class="feature-card"><h3 style="color:#38BDF8">📚 Vector RAG</h3><p style="color:#94A3B8">Semantic search through documents using ChromaDB.</p></div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="feature-card"><h3 style="color:#818CF8">🎯 Patterns</h3><p style="color:#94A3B8">AI-driven exam trend analysis and question forecasting.</p></div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="feature-card"><h3 style="color:#F472B6">💬 AI Tutor</h3><p style="color:#94A3B8">Persistent chat memory for deep conceptual doubts.</p></div>', unsafe_allow_html=True)

# --- 6. ADVANCED RAG LOGIC ---
if submit and uploaded_file:
    with st.spinner("🧠 Processing Knowledge Base..."):
        try:
            text_data = ""
            # Handle PDF Ingestion
            if uploaded_file.type == "application/pdf":
                text_data = get_pdf_text(uploaded_file)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text_data)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.vector_db = Chroma.from_texts(chunks, embeddings)
            
            # Initial Report Generation
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            prompt_header = f"Acting as a {mood} expert, "
            
            if doc_type == "Notes Generation":
                task_prompt = f"{prompt_header} generate unit-wise detailed notes. Use '---SUMMARY---' and '---QUIZ---' as markers."
            else:
                task_prompt = f"{prompt_header} analyze these PYQs for trends. Use '---SUMMARY---' and '---QUIZ---' as markers."

            if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                img_b64 = encode_image(uploaded_file)
                msg = HumanMessage(content=[{"type": "text", "text": task_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}])
                res = model.invoke([msg])
            else:
                res = model.invoke(f"{task_prompt}\n\nContent: {text_data}")
            
            st.session_state.initial_analysis = res.content
            st.success("✅ Training Complete!")

        except Exception as e:
            st.error(f"Error: {e}")

# --- 7. DISPLAY & INTERACTIVE CHAT ---
if st.session_state.initial_analysis:
    analysis = st.session_state.initial_analysis
    t1, t2, t3 = st.tabs(["📑 Report", "⚡ Revision", "🧠 Quiz"])
    
    with t1:
        st.markdown(f'<div class="content-card">{analysis.split("---SUMMARY---")[0]}</div>', unsafe_allow_html=True)
    with t2:
        if "---SUMMARY---" in analysis:
            st.markdown(f'<div class="content-card">{analysis.split("---SUMMARY---")[1].split("---QUIZ---")[0]}</div>', unsafe_allow_html=True)
    with t3:
        if "---QUIZ---" in analysis:
            st.markdown(f'<div class="content-card">{analysis.split("---QUIZ---")[1]}</div>', unsafe_allow_html=True)

    # MODERN CHAT INTERFACE
    st.markdown("---")
    st.markdown("### 💬 Chat with your AI Tutor")
    
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    if query := st.chat_input("Ask a follow-up question..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching Knowledge Base..."):
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                
                # Manual RAG Retrieval (Stable & Faster)
                context = ""
                if st.session_state.vector_db:
                    docs = st.session_state.vector_db.similarity_search(query, k=3)
                    context = "\n".join([d.page_content for d in docs])
                
                # Building the memory-aware prompt
                chat_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
                
                final_prompt = f"""
                You are a helpful AI Study Tutor. Answer based on the Context provided.
                
                Previous Conversation:
                {chat_context}
                
                Context from Document:
                {context}
                
                Current Question: {query}
                """
                
                response = model.invoke(final_prompt)
                st.markdown(response.content)
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                st.rerun()

st.markdown("<br><hr><center style='color:#64748B;'>Mayank's AI Lab | Multimodal RAG 2026</center>", unsafe_allow_html=True)