import streamlit as st
import os
import base64
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image
from dotenv import load_dotenv

# --- 1. CONFIG & ELITE UI STYLING ---
load_dotenv()
st.set_page_config(page_title="StudyBuddy AI Pro", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    /* Global Reset */
    .stApp {
        background-color: #0B0F1A;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #F8FAFC;
    }

    /* Hide Streamlit Default Menus for Professional Look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Sidebar - Deep Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937;
    }

    /* Professional Top Branding (Inside Sidebar) */
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

    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 80px 20px 40px 20px;
        background: radial-gradient(circle at center, rgba(37, 99, 235, 0.1) 0%, transparent 70%);
    }

    .hero-title {
        font-size: 4.5rem !important;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -2px;
        line-height: 1;
    }

    /* Feature Cards */
    .feature-card {
        background: #1E293B;
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid #334155;
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        border-color: #38BDF8;
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    /* Modern Button */
    .stButton>button {
        width: 100%;
        border-radius: 14px;
        height: 3.8rem;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white !important;
        font-weight: 700;
        border: none;
        transition: 0.3s;
    }

    /* Output Card */
    .content-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid #334155;
        backdrop-filter: blur(12px);
        margin-top: 20px;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
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

# --- 2. SETUP API KEY ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🔑 API Key not found! Please check your .env file.")
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

# --- 4. SIDEBAR (Control Center) ---
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="brand-text">STUDYBUDDY AI</div></div>', unsafe_allow_html=True)
    
    doc_type = st.radio("⚡ Operation Mode", ["Notes Generation", "Exam Prediction"])
    mood = st.selectbox("🎭 AI Teaching Style", ["Academic", "Explain Like I'm 5", "Interview Prep"])
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    submit = st.button("🚀 Analyze & Generate")
    st.caption("Mayank's AI Lab | v2.5 Pro Edition")

# --- 5. MAIN CONTENT ---
if not submit:
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Academic <span style="color:#38BDF8">Intelligence.</span></h1>
            <p style="color:#94A3B8; font-size:1.2rem; max-width:700px; margin:auto;">
                The most advanced Multimodal AI Agent for students. Turn raw syllabus and question papers into structured study kits instantly.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><h3 style="color:#38BDF8">📚 Smart Notes</h3><p style="color:#94A3B8">Unit-wise notes with priority tags from handwriting or PDFs.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><h3 style="color:#818CF8">🎯 Patterns</h3><p style="color:#94A3B8">Analyze exam trends and predict likely 10-mark questions.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><h3 style="color:#F472B6">🧠 Recall</h3><p style="color:#94A3B8">Self-test your knowledge with AI-generated test series.</p></div>', unsafe_allow_html=True)

# --- 6. AGENT LOGIC ---
if submit and uploaded_file:
    with st.spinner("Processing with Gemini 2.5 Flash..."):
        try:
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            
            prompt_text = f"Acting as an {mood}, "
            if doc_type == "Notes Generation":
                prompt_text += "create detailed notes from this. Use '---SUMMARY---' and '---QUIZ---' as separators."
            else:
                prompt_text += "analyze these PYQs for trends. Use '---SUMMARY---' and '---QUIZ---' as separators."

            if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                base64_image = encode_image(uploaded_file)
                message = HumanMessage(content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
                response = model.invoke([message])
            else:
                raw_text = get_pdf_text(uploaded_file)
                response = model.invoke(f"{prompt_text}\n\nContent: {raw_text}")
            
            full_text = response.content
            
            # Parsing and Tabs
            tab1, tab2, tab3 = st.tabs(["📑 Intelligence", "⚡ Cheat Sheet", "🧠 Practice"])
            
            with tab1:
                st.markdown(f'<div class="content-card">{full_text.split("---SUMMARY---")[0]}</div>', unsafe_allow_html=True)
                st.download_button("Download Report", full_text, "StudyBuddy_Report.txt")
            with tab2:
                if "---SUMMARY---" in full_text:
                    st.markdown(f'<div class="content-card">{full_text.split("---SUMMARY---")[1].split("---QUIZ---")[0]}</div>', unsafe_allow_html=True)
            with tab3:
                if "---QUIZ---" in full_text:
                    st.markdown(f'<div class="content-card">{full_text.split("---QUIZ---")[1]}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Agent failed: {e}")

st.markdown("<br><hr><center style='color:#64748B;'>Engineered by Mayank | 2026</center>", unsafe_allow_html=True)