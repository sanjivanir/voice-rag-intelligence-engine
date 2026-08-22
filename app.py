import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from harness import run_rag_pipeline, speech_to_text
from ingest import get_msmarco_passages, sentence_chunking
from retriever import build_index

load_dotenv()

st.set_page_config(
    page_title="Voice RAG Intelligence Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def initialize_knowledge_base():
    raw_passages = get_msmarco_passages(limit=100)
    chunks = sentence_chunking(raw_passages)
    build_index(chunks, force_rebuild=True)
    return len(chunks)

total_chunks_indexed = initialize_knowledge_base()

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .header-box {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .badge {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #0284C7;
        margin-right: 8px;
    }
    .card-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-box">
    <span class="badge">Sarvam AI STT</span>
    <span class="badge">Qdrant Index: {total_chunks_indexed} Chunks</span>
    <span class="badge">Groq Llama 3</span>
    <h1 style="color: #F8FAFC; margin-top: 12px; margin-bottom: 0px;">🎙️ Voice RAG Intelligence Engine</h1>
    <p style="color: #94A3B8; margin-top: 6px; font-size: 1.05rem;">
        Ultra-low latency speech-driven retrieval augmented generation.
    </p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Select Audio Query")
    tab1, tab2 = st.tabs(["📁 File Upload", "🎙️ Live Microphone"])
    
    recorded_audio = None
    uploaded_audio = None
    
    with tab1:
        st.caption("Upload an audio query file (.wav, .mp3, .m4a, .aac).")
        uploaded_audio = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a", "aac"], label_visibility="collapsed")

    with tab2:
        st.caption("Record voice input directly from your browser.")
        recorded_audio = st.audio_input("Record Question")

active_audio = uploaded_audio or recorded_audio

with col_right:
    st.subheader("2. Pipeline Diagnostics & Response")
    
    if active_audio is not None:
        st.audio(active_audio)
        
        if st.button("⚡ Execute Voice RAG Pipeline", type="primary", use_container_width=True):
            total_start_time = time.time()
            
            if hasattr(active_audio, 'name') and '.' in active_audio.name:
                file_ext = "." + active_audio.name.split(".")[-1].lower()
            else:
                file_ext = ".wav"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(active_audio.read())
                tmp_path = tmp_file.name

            # Step 1: Speech-to-Text
            with st.spinner("Transcribing audio via Sarvam AI..."):
                stt_start = time.time()
                try:
                    transcript = speech_to_text(tmp_path)
                    stt_duration = (time.time() - stt_start) * 1000
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if not transcript or str(transcript).startswith("Sarvam Error") or str(transcript).startswith("ERROR:") or str(transcript).startswith("STT Exception:"):
                st.error(f"STT Diagnostic Output: {transcript}")
            else:
                # Step 2: RAG Pipeline Execution
                with st.spinner("Searching vector database & generating answer..."):
                    rag_start = time.time()
                    answer = run_rag_pipeline(transcript)
                    rag_duration = (time.time() - rag_start) * 1000

                total_duration = (time.time() - total_start_time) * 1000

                # Latency metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("STT Latency", f"{stt_duration:.0f} ms")
                m2.metric("RAG Engine Latency", f"{rag_duration:.0f} ms")
                m3.metric("Total E2E Speed", f"{total_duration:.0f} ms")

                st.markdown(f"""
                <div class="card-box" style="border-left: 4px solid #38BDF8;">
                    <div style="color: #94A3B8; font-size: 0.85rem; font-weight: bold;">SPEECH-TO-TEXT TRANSCRIPT</div>
                    <div style="color: #F8FAFC; font-size: 1.1rem; margin-top: 4px;">"{transcript}"</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="card-box" style="border-left: 4px solid #10B981;">
                    <div style="color: #94A3B8; font-size: 0.85rem; font-weight: bold;">GENERATED RAG ANSWER</div>
                    <div style="color: #F8FAFC; font-size: 1.05rem; margin-top: 6px; line-height: 1.5;">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Upload or record an audio file to run the pipeline.")