import warnings
import logging

# 1. SILENCE TERMINAL WARNINGS (Must be at the absolute top)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)

import streamlit as st
import os
import base64
import re
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv

# Modern LangChain Ecosystem Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- 2. CONFIG & ELITE UI STYLING ---
load_dotenv()
st.set_page_config(
    page_title="StudyBuddy AI Pro", 
    layout="wide", 
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

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
    
    /* Clean UI Footers */
    footer {visibility: hidden;} #MainMenu {visibility: hidden;}
    
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

# --- 3. API SETUP ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🔑 API Key Missing! Ensure your .env file has GOOGLE_API_KEY.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 4. DATA SANITIZATION UTILITIES ---
def clean_extracted_text(text):
    """
    Advanced multi-pass pre-processing sanitizer designed to remove metadata noise,
    garbled alphanumeric anomalies, and structural layout headers from handwriting datasets.
    """
    if not text:
        return ""
    # Pass 1: Clear obvious branding watermarks or recurrent page line items
    text = re.sub(r'(?i)\b(classmate|date|page|source|\d+\.?)\b', '', text)
    
    # Pass 2: Purge garbled OCR artifact patterns (isolated clusters of corrupt symbols/letters)
    text = re.sub(r'\b[a-zA-Z0-9åß•æ«œ»„\-\+\.\,]{10,}\b', '', text)
    
    # Pass 3: Re-group whitespace line fragments
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_lines = [l for l in lines if len(re.findall(r'[a-zA-Z]', l)) > 2] # Drops lines with mostly numbers/junk symbols
    
    return "\n".join(clean_lines)

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def get_pdf_text(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        content = page.extract_text()
        if content: 
            text += content
    return text

# --- 5. SIDEBAR ---
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

# --- 6. HERO INTERFACE (Pre-Upload) ---
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

# --- 7. AGENT & RAG LOGIC ---
if submit and uploaded_file:
    with st.spinner("🧠 Sanitizing Input & Mapping Knowledge Base..."):
        try:
            raw_text = ""
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            
            if uploaded_file.type == "application/pdf":
                raw_text = get_pdf_text(uploaded_file)
                # Pass data through the sanitization engine
                text_for_db = clean_extracted_text(raw_text)
                
                if not text_for_db.strip():
                    text_for_db = raw_text # Fallback if text is too sparse
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text_for_db)
                st.session_state.vector_db = Chroma.from_texts(chunks, embeddings)
            else:
                text_for_db = "Visual input detected. Extracting core technical concepts directly via vision model..."
                st.session_state.vector_db = Chroma.from_texts([text_for_db], embeddings)

            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            
            formatting_instructions = (
                "\n\nCRITICAL STRATIFICATION RULES:\n"
                "You MUST divide your entire output using these exact flags placed on their own empty line:\n"
                "---SUMMARY---\n"
                "---QUIZ---\n\n"
                "Section Directives:\n"
                "1. Section 1 (Detailed Notes): Act as an expert University Professor in Computer Science. "
                "You are reviewing handwritten, raw, or unstructured text inputs. Look past any residual line anomalies, "
                "garbled symbols, metadata labels, or layout text. Target the true structural core concepts (e.g., Programming Logic, "
                "Variables, Data Structures, Conditional checks, Core syntax paths). Reconstruct and extrapolate these topics into "
                "highly comprehensive, beautiful, long university textbook-quality study notes. Include code layout parameters and full "
                "technical narratives. "
                "Crucial: Explicitly look for areas where visual aids explain the concept best and inject block placeholders like "
                "'[🖼️ DIAGRAM TIP: Insert structural chart here]' right inside the notes text.\n"
                "2. Section 2 (Fast Revision): Provide a high-density, beautifully arranged 1-page summary cheat sheet focusing on core syntax declarations, expressions, and vital operators.\n"
                "3. Section 3 (Quiz): Provide exactly 5 distinct conceptual questions numbered 1 to 5: 2 MCQs (with choice labels A, B, C, D) and 3 Theoretical short-answer questions. Do not include introductory descriptions or formatting tables after the quiz flag line."
            )

            if doc_type == "Notes Generation":
                task = f"Extract the substantial technical concepts from the text dataset and expand them into professional textbook notes. Ignore noisy characters or template metadata completely. {formatting_instructions}"
            else:
                task = f"Analyze historical query trends and pattern structures inside this paper to compile predicted 10-mark examination topics. {formatting_instructions}"

            if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                img_b64 = encode_image(uploaded_file)
                msg = HumanMessage(content=[{"type": "text", "text": task}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}])
                res = model.invoke([msg])
            else:
                res = model.invoke(f"{task}\n\nContent:\n{text_for_db}")
            
            st.session_state.initial_analysis = res.content
            st.success("✅ Knowledge Base Initialized Successfully!")

        except Exception as e:
            st.error(f"Logic Error: {e}")

# --- 8. DISPLAY RESULTS & INTERACTIVE COMPONENTS ---
if st.session_state.initial_analysis:
    analysis = st.session_state.initial_analysis
    
    parts_summary = re.split(r'(?i)---SUMMARY---', analysis)
    notes_content = parts_summary[0].strip()
    
    summary_content = ""
    quiz_content = ""
    
    if len(parts_summary) > 1:
        parts_quiz = re.split(r'(?i)---QUIZ---', parts_summary[1])
        summary_content = parts_quiz[0].strip()
        if len(parts_quiz) > 1:
            quiz_content = parts_quiz[1].strip()

    t1, t2, t3 = st.tabs(["📑 Detailed Notes", "⚡ Fast Revision", "🧠 Interactive Quiz"])
    
    with t1:
        st.markdown(f'<div class="content-card">{notes_content}</div>', unsafe_allow_html=True)
    
    with t2:
        if summary_content:
            st.markdown(f'<div class="content-card">{summary_content}</div>', unsafe_allow_html=True)
        else:
            st.info("No summary content detected. Try generating the content again.")
    
    with t3:
        if quiz_content:
            st.markdown("### 📝 Active Recall Challenge (MCQs & Theoretical)")
            
            quiz_lines = [line.strip() for line in quiz_content.split('\n') if line.strip()]
            quiz_blocks = []
            current_block = []
            
            for line in quiz_lines:
                if re.match(r'^(?:Q?\d+[:\.)]|\d+\.)', line) and current_block:
                    quiz_blocks.append("\n".join(current_block))
                    current_block = [line]
                else:
                    current_block.append(line)
            if current_block:
                quiz_blocks.append("\n".join(current_block))
            
            # Drops structural markdown headers, intro artifacts, and empty padding string splits completely
            clean_questions = [
                b for b in quiz_blocks 
                if len(b.strip()) > 15 
                and not b.lower().startswith("instructions") 
                and not b.lower().startswith("here is")
                and not b.lower().startswith("choose the")
            ]

            if not clean_questions:
                st.info("💡 Quiz compilation buffer caught. Please hit 'Analyze & Train Agent' again to compile.")
            else:
                for i, block in enumerate(clean_questions[:5]):
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    st.write(f"**Question {i+1}:**")
                    st.markdown(block)
                    
                    u_ans = st.text_input(f"Type your answer or selection for Q{i+1}:", key=f"q_{i}")
                    
                    if st.button(f"Submit & Grade Answer {i+1}", key=f"b_{i}"):
                        if u_ans:
                            with st.spinner("Evaluating performance..."):
                                eval_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
                                eval_res = eval_model.invoke(
                                    f"Question: {block}\n"
                                    f"Student Answer: {u_ans}\n\n"
                                    f"Evaluate the student's answer accurately. If their understanding or selection matches the correct concept, begin your response with exactly 'CORRECT'. If it is incorrect or missing vital definitions, start with exactly 'WRONG'. Follow with a clear explanation."
                                )
                                
                                evaluation = eval_res.content.strip()
                                
                                if evaluation.upper().startswith("CORRECT"):
                                    st.success("🎉 Correct! Fantastic job tracking the core concepts!")
                                    st.info(evaluation[7:].strip())
                                else:
                                    st.error("❌ Wrong! Try again.")
                                    st.warning(evaluation[5:].strip() if evaluation.upper().startswith("WRONG") else evaluation)
                        else:
                            st.warning("Please type your response before submission.")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No active recall quiz data detected. Try re-running the generator.")

    # 💬 CHATBOT INTERFACE (Persistent Memory RAG)
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
                ctx = ""
                if st.session_state.vector_db:
                    hits = st.session_state.vector_db.similarity_search(query, k=3)
                    ctx = "\n".join([h.page_content for h in hits])
                
                history_window = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-4:]])
                final_p = f"Context:\n{ctx}\n\nHistory:\n{history_window}\n\nQuestion: {query}\n\nAnswer professionally:"
                
                ai_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                ai_ans = ai_model.invoke(final_p)
                
                st.markdown(ai_ans.content)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_ans.content})
                st.rerun()

st.markdown("<br><hr><center style='color:#64748B;'>Engineered by Mayank | 2026 Pro Edition</center>", unsafe_allow_html=True)