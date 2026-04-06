import streamlit as st
import os
import base64
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv

# Modern LangChain Ecosystem Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- 1. CONFIG & ELITE UI STYLING ---
load_dotenv()
st.set_page_config(page_title="StudyBuddy AI Pro", layout="wide", page_icon="🎓")

# Persistent State Management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "initial_analysis" not in st.session_state:
    st.session_state.initial_analysis = None

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    /* Global Background & Font */
    .stApp { 
        background-color: #0B0F1A; 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        color: #F8FAFC; 
    }
    
    /* Hide Menus */
    header {visibility: hidden;} footer {visibility: hidden;} #MainMenu {visibility: hidden;}
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] { 
        background-color: #111827 !important; 
        border-right: 1px solid #1F2937; 
    }
    
    .sidebar-brand { 
        padding: 1.5rem; 
        text-align: center; 
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(129, 140, 248, 0.1)); 
        border-radius: 16px; 
        margin-bottom: 2rem; 
        border: 1px solid rgba(56, 189, 248, 0.2); 
    }
    
    .brand-text { 
        font-size: 1.5rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, #38BDF8, #818CF8); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
    }

    /* Hero Section Styling */
    .hero-section { 
        text-align: center; 
        padding: 60px 20px 30px 20px; 
        background: radial-gradient(circle at center, rgba(37, 99, 235, 0.1) 0%, transparent 70%); 
    }
    
    .hero-title { 
        font-size: 4rem !important; 
        font-weight: 800; 
        letter-spacing: -2px; 
    }

    /* Elite Cards */
    .feature-card { 
        background: #1E293B; 
        padding: 2rem; 
        border-radius: 24px; 
        border: 1px solid #334155; 
        height: 100%; 
    }
    
    .content-card { 
        background: rgba(30, 41, 59, 0.7); 
        padding: 2rem; 
        border-radius: 24px; 
        border: 1px solid #334155; 
        backdrop-filter: blur(12px); 
        margin-top: 15px; 
        color: #E2E8F0; 
    }

    /* Buttons & Tabs */
    .stButton>button { 
        width: 100%; 
        border-radius: 14px; 
        height: 3.5rem; 
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); 
        color: white !important; 
        font-weight: 700; 
        border: none; 
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1E293B; 
        border-radius: 12px; 
        padding: 10px 20px; 
        color: #94A3B8; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #3B82F6 !important; 
        color: white !important; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. API SETUP ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🔑 API Key Missing! Ensure your .env file has GOOGLE_API_KEY.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 3. HELPER FUNCTIONS ---
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
    uploaded_file = st.file_uploader("📂 Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    submit = st.button("🚀 Analyze & Train Agent")
    
    if st.button("🗑️ Reset Application"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.session_state.initial_analysis = None
        st.rerun()

# --- 5. HERO INTERFACE (Pre-Upload) ---
if not st.session_state.initial_analysis:
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Academic <span style="color:#38BDF8">Intelligence.</span></h1>
            <p style="color:#94A3B8; font-size:1.2rem; max-width:700px; margin:auto;">
                The modern way to study. Turn raw documents into persistent knowledge bases and master concepts through AI-led active recall.
            </p>
        </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="feature-card"><h3 style="color:#38BDF8">📚 Vector RAG</h3><p style="color:#94A3B8">Semantic indexing for perfect conceptual retrieval.</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="feature-card"><h3 style="color:#818CF8">🎯 Predictor</h3><p style="color:#94A3B8">Forecasting exam trends based on historical patterns.</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="feature-card"><h3 style="color:#F472B6">💬 AI Tutor</h3><p style="color:#94A3B8">Context-aware doubts resolution with persistent memory.</p></div>', unsafe_allow_html=True)

# --- 6. AGENT & RAG LOGIC ---
if submit and uploaded_file:
    with st.spinner("🧠 Initializing Knowledge Base & Training Tutor..."):
        try:
            text_for_db = ""
            # Handle PDF
            if uploaded_file.type == "application/pdf":
                text_for_db = get_pdf_text(uploaded_file)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text_for_db)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.vector_db = Chroma.from_texts(chunks, embeddings)
            else:
                text_for_db = "Visual input detected. Analyzing multimodal context..."

            # Initial Generation
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            prompt_header = f"Acting as a {mood} expert, "
            if doc_type == "Notes Generation":
                task = f"{prompt_header} create detailed unit-wise notes. Use '---SUMMARY---' and '---QUIZ---' as separators."
            else:
                task = f"{prompt_header} predict likely exam questions based on these PYQs. Use '---SUMMARY---' and '---QUIZ---' as separators."

            if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                img_b64 = encode_image(uploaded_file)
                msg = HumanMessage(content=[{"type": "text", "text": task}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}])
                res = model.invoke([msg])
            else:
                res = model.invoke(f"{task}\n\nContent: {text_for_db}")
            
            st.session_state.initial_analysis = res.content
            st.success("✅ Tutor Trained Successfully!")

        except Exception as e:
            st.error(f"Logic Error: {e}")

# --- 7. DISPLAY RESULTS & INTERACTIVE COMPONENTS ---
if st.session_state.initial_analysis:
    analysis = st.session_state.initial_analysis
    t1, t2, t3 = st.tabs(["📑 Detailed Intelligence", "⚡ Fast Revision", "🧠 Interactive Quiz"])
    
    with t1:
        st.markdown(f'<div class="content-card">{analysis.split("---SUMMARY---")[0]}</div>', unsafe_allow_html=True)
    
    with t2:
        if "---SUMMARY---" in analysis:
            st.markdown(f'<div class="content-card">{analysis.split("---SUMMARY---")[1].split("---QUIZ---")[0]}</div>', unsafe_allow_html=True)
    
    with t3:
        if "---QUIZ---" in analysis:
            st.markdown("### 📝 Active Recall Challenge")
            quiz_text = analysis.split("---QUIZ---")[1].strip()
            quiz_blocks = [b.strip() for b in quiz_text.split('\n\n') if b.strip()]
            
            for i, block in enumerate(quiz_blocks[:5]):
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.write(f"**Q{i+1}:** {block.split('Correct:')[0]}")
                
                u_ans = st.text_input(f"Your explanation for Q{i+1}:", key=f"q_{i}")
                if st.button(f"Grade Answer {i+1}", key=f"b_{i}"):
                    if u_ans:
                        eval_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                        eval_res = eval_model.invoke(f"Question: {block}\nUser Answer: {u_ans}\nGrade this and explain clearly.")
                        st.info(eval_res.content)
                    else:
                        st.warning("Please type an answer first.")
                st.markdown('</div>', unsafe_allow_html=True)

    # 💬 CHATBOT INTERFACE (Persistent Memory)
    st.markdown("---")
    st.markdown("### 💬 Concept Doubts? Chat with StudyBuddy")
    
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    if query := st.chat_input("Ask about any specific topic from your file..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving Knowledge..."):
                # RAG Retrieval
                ctx = ""
                if st.session_state.vector_db:
                    hits = st.session_state.vector_db.similarity_search(query, k=3)
                    ctx = "\n".join([h.page_content for h in hits])
                
                # Chat logic with memory
                history_window = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-4:]])
                final_p = f"Context:\n{ctx}\n\nHistory:\n{history_window}\n\nQuestion: {query}\n\nAnswer professionally:"
                
                ai_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                ai_ans = ai_model.invoke(final_p)
                
                st.markdown(ai_ans.content)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_ans.content})
                st.rerun()

st.markdown("<br><hr><center style='color:#64748B;'>Engineered by Mayank | 2026 Pro Edition</center>", unsafe_allow_html=True)