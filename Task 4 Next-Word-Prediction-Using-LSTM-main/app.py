import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import time
from datetime import datetime
import json
import os
from gtts import gTTS
import io

# Page configuration
st.set_page_config(
    page_title="AI Word Predictor | Next Word Prediction",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Fix text visibility */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, div, span {
        color: #1a1a1a !important;
    }
    
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    
    .stMetric label {
        color: #666 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #6366f1 !important;
    }
    
    .title-container {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .feature-card h3 {
        color: #6366f1 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #555 !important;
        margin: 0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-size: 1.3rem;
        line-height: 1.8;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
        min-height: 200px;
        margin: 1rem 0;
    }
    
    .suggestion-chip {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(99, 102, 241, 0.3);
    }
    
    .suggestion-chip:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.35);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem;
        font-size: 1rem;
        color: #1a1a1a !important;
        background-color: white !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        color: #1a1a1a !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999 !important;
    }
    
    .sidebar .sidebar-content {
        background: white;
    }
    
    .history-item {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #6366f1;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-item strong {
        color: #6366f1 !important;
    }
    
    .history-item small {
        color: #666 !important;
    }
    
    .history-item:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .loading-spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid #6366f1;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .metric-card h3, .metric-card p {
        color: #1a1a1a !important;
    }
    
    /* Info and success messages */
    .stInfo, .stSuccess, .stWarning, .stError {
        color: #1a1a1a !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] {
        color: #1a1a1a !important;
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #6366f1 !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
        color: #1a1a1a !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #666 !important;
    }
    
    /* Button text */
    .stButton > button {
        color: white !important;
    }
    
    /* Selectbox and slider text */
    .stSelectbox, .stSlider {
        color: #1a1a1a !important;
    }
    
    /* Fix dropdown/selectbox visibility */
    .stSelectbox > div > div > select {
        color: #1a1a1a !important;
        background-color: white !important;
    }
    
    .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown options - fix black background */
    div[data-baseweb="select"] {
        color: #1a1a1a !important;
    }
    
    div[data-baseweb="select"] > div {
        color: #1a1a1a !important;
        background-color: white !important;
    }
    
    /* Dropdown menu container - white background */
    div[data-baseweb="popover"] {
        background-color: white !important;
    }
    
    div[data-baseweb="popover"] > div {
        background-color: white !important;
    }
    
    /* Dropdown menu items */
    ul[role="listbox"] {
        background-color: white !important;
    }
    
    ul[role="listbox"] li {
        color: #1a1a1a !important;
        background-color: white !important;
    }
    
    ul[role="listbox"] li:hover {
        background-color: #f1f5f9 !important;
        color: #6366f1 !important;
    }
    
    /* Streamlit selectbox dropdown */
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
    }
    
    /* Fix any black backgrounds in dropdown */
    [data-baseweb="menu"] {
        background-color: white !important;
    }
    
    [data-baseweb="menu"] > div {
        background-color: white !important;
    }
    
    /* Checkbox label */
    .stCheckbox label {
        color: #1a1a1a !important;
    }
    
    .progress-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        height: 100%;
        transition: width 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer with caching
@st.cache_resource
def load_resources():
    try:
        model = load_model("my_model1.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

model, tokenizer, model_loaded = load_resources()

# Initialize session state
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'word_count' not in st.session_state:
    st.session_state.word_count = 0
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'quick_start_text' not in st.session_state:
    st.session_state.quick_start_text = None
if 'current_seed_text' not in st.session_state:
    st.session_state.current_seed_text = ""

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h2 style='color: #6366f1; margin-bottom: 2rem;'>⚙️ Settings</h2>
        </div>
    """, unsafe_allow_html=True)
    
    max_words = st.slider("Words to Generate", 1, 50, 20, 1)
    prediction_mode = st.selectbox(
        "Prediction Mode",
        ["Top Prediction", "Top 3 Suggestions", "Top 5 Suggestions"]
    )
    animation_speed = st.slider("Animation Speed", 0.1, 1.0, 0.3, 0.1)
    show_probabilities = st.checkbox("Show Probabilities", value=False)
    enable_audio = st.checkbox("🔊 Generate Audio", value=True, help="Generate text-to-speech audio automatically")
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Words Generated", st.session_state.word_count)
    with col2:
        st.metric("Predictions Made", st.session_state.prediction_count)
    
    st.markdown("---")
    
    st.markdown("### 📝 History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            with st.container():
                st.markdown(f"""
                    <div class='history-item' onclick='selectHistory({i})'>
                        <strong>{item['timestamp']}</strong><br>
                        <small>{item['text'][:50]}...</small>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No history yet. Start generating text!")
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Main content
st.markdown("""
    <div class='title-container'>
        <h1 class='main-title'>✨ AI Word Predictor</h1>
        <p class='subtitle'>Powered by LSTM Neural Networks | Predict the next word with AI</p>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model could not be loaded. Please ensure 'my_model.h5' and 'tokenizer.pkl' are in the directory.")
    st.stop()

# Main input area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Your Text")
    
    # Use quick start text if available, otherwise use current seed text or empty
    if st.session_state.quick_start_text:
        default_text = st.session_state.quick_start_text
        st.session_state.current_seed_text = st.session_state.quick_start_text
        st.session_state.quick_start_text = None  # Clear after using
    elif st.session_state.current_seed_text:
        default_text = st.session_state.current_seed_text
    else:
        default_text = ""
    
    seed_text = st.text_input(
        "Start typing or enter your seed text:",
        value=default_text,
        placeholder="Type your starting phrase here...",
        key="seed_input"
    )
    
    # Update current seed text when user types
    if seed_text:
        st.session_state.current_seed_text = seed_text
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn1:
        generate_btn = st.button("🚀 Generate", use_container_width=True, type="primary")
    
    with col_btn2:
        predict_next_btn = st.button("🔮 Predict Next", use_container_width=True)
    
    with col_btn3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

with col2:
    st.markdown("### 💡 Quick Start")
    quick_starts = [
        "The Count said",
        "Welcome to my",
        "It was on the",
        "I found that my"
    ]
    for quick_text in quick_starts:
        if st.button(quick_text, key=f"quick_{quick_text}", use_container_width=True):
            st.session_state.quick_start_text = quick_text
            st.rerun()

if clear_btn:
    st.session_state.generated_text = ""
    st.session_state.audio_bytes = None
    st.session_state.current_seed_text = ""
    st.rerun()

# Prediction functions
def predict_next_word(text, top_n=5):
    """Predict next word(s) with probabilities"""
    try:
        tokenized_text = tokenizer.texts_to_sequences([text])[0]
        if not tokenized_text:
            return []
        
        padded_text = pad_sequences([tokenized_text], maxlen=16, padding='pre')
        predictions = model.predict(padded_text, verbose=0)[0]
        
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        
        results = []
        for idx in top_indices:
            if idx in reverse_word_index:
                word = reverse_word_index[idx]
                prob = predictions[idx]
                results.append((word, prob))
        
        return results
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

def generate_text(seed_text, num_words, show_animation=True, generate_audio=True):
    """Generate text with optional animation and parallel audio generation"""
    import threading
    
    text = seed_text
    output_placeholder = st.empty()
    audio_placeholder = st.empty()
    
    if show_animation:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Function to generate audio in background
    def generate_audio_background(text_to_speak):
        try:
            tts = gTTS(text=text_to_speak, lang='en', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.session_state.audio_bytes = audio_fp
        except Exception as e:
            st.session_state.audio_bytes = None
    
    # Start audio generation in background thread after collecting some words
    audio_thread = None
    words_for_audio = 5  # Start generating audio after 5 words
    
    for i in range(num_words):
        try:
            tokenized_text = tokenizer.texts_to_sequences([text])[0]
            if not tokenized_text:
                break
            
            padded_text = pad_sequences([tokenized_text], maxlen=16, padding='pre')
            pos = np.argmax(model.predict(padded_text, verbose=0))
            
            word_found = False
            for word, index in tokenizer.word_index.items():
                if index == pos:
                    text = text + " " + word
                    word_found = True
                    break
            
            if not word_found:
                break
            
            # Start audio generation in parallel after collecting some words
            if generate_audio and i == words_for_audio and audio_thread is None:
                audio_thread = threading.Thread(target=generate_audio_background, args=(text,))
                audio_thread.daemon = True
                audio_thread.start()
                audio_placeholder.info("🔊 Generating audio in background...")
            
            if show_animation:
                progress = (i + 1) / num_words
                progress_bar.progress(progress)
                status_text.text(f"Generating... {i+1}/{num_words} words")
                
                output_placeholder.markdown(f"""
                    <div class='prediction-box'>
                        {text}
                    </div>
                """, unsafe_allow_html=True)
                
                time.sleep(animation_speed)
            else:
                output_placeholder.markdown(f"""
                    <div class='prediction-box'>
                        {text}
                    </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            break
    
    # Generate final audio with complete text if audio is enabled
    if generate_audio:
        # Wait for background thread if it's still running
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=1)
        
        # Generate final audio with complete text
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.session_state.audio_bytes = audio_fp
            audio_placeholder.empty()
        except Exception as e:
            st.session_state.audio_bytes = None
            audio_placeholder.warning(f"Audio generation failed: {str(e)}")
    else:
        st.session_state.audio_bytes = None
        audio_placeholder.empty()
    
    if show_animation:
        progress_bar.empty()
        status_text.empty()
    
    st.session_state.generated_text = text
    st.session_state.word_count += num_words
    st.session_state.prediction_count += num_words
    
    # Save to history
    st.session_state.history.append({
        'text': text,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'seed': seed_text,
        'words': num_words
    })
    
    return text

# Handle button clicks
if generate_btn and seed_text:
    with st.spinner("Generating text..."):
        generated = generate_text(seed_text, max_words, show_animation=True, generate_audio=enable_audio)
        st.success("✅ Text generated successfully!")

elif predict_next_btn and seed_text:
    st.session_state.prediction_count += 1
    top_n = {"Top Prediction": 1, "Top 3 Suggestions": 3, "Top 5 Suggestions": 5}[prediction_mode]
    
    predictions = predict_next_word(seed_text, top_n=top_n)
    
    if predictions:
        st.markdown("### 🔮 Predictions")
        
        cols = st.columns(min(top_n, 5))
        for idx, (word, prob) in enumerate(predictions):
            with cols[idx % len(cols)]:
                if show_probabilities:
                    st.markdown(f"""
                        <div class='suggestion-chip' style='text-align: center; padding: 1rem;'>
                            <strong>{word}</strong><br>
                            <small>{prob*100:.1f}%</small>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='suggestion-chip' style='text-align: center; padding: 1rem;'>
                            <strong>{word}</strong>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Show selected word
        selected_word = predictions[0][0]
        new_text = seed_text + " " + selected_word
        st.markdown(f"""
            <div class='prediction-box' style='margin-top: 1rem;'>
                <strong>Suggested continuation:</strong><br>
                {new_text}
            </div>
        """, unsafe_allow_html=True)

# Display generated text
if st.session_state.generated_text:
    st.markdown("### ✨ Generated Text")
    st.markdown(f"""
        <div class='prediction-box'>
            {st.session_state.generated_text}
        </div>
    """, unsafe_allow_html=True)
    
    # Audio player
    if st.session_state.audio_bytes:
        st.markdown("### 🔊 Audio")
        st.audio(st.session_state.audio_bytes, format='audio/mp3')
        if enable_audio:
            st.caption("🎵 Text-to-Speech audio generated automatically")
        else:
            st.caption("🎵 Audio was generated previously")
    elif enable_audio:
        st.info("💡 Audio generation is enabled. Generate text to create audio.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Copy Text", use_container_width=True):
            st.write(st.session_state.generated_text)
            st.success("✅ Copied to clipboard!")
    
    with col2:
        if st.button("💾 Save Text", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_text_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(st.session_state.generated_text)
            st.success(f"✅ Saved as {filename}")
    
    with col3:
        if st.button("🔊 Regenerate Audio", use_container_width=True, disabled=not enable_audio):
            if enable_audio:
                try:
                    with st.spinner("Generating audio..."):
                        tts = gTTS(text=st.session_state.generated_text, lang='en', slow=False)
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0)
                        st.session_state.audio_bytes = audio_fp
                        st.success("✅ Audio regenerated!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Audio generation failed: {str(e)}")
            else:
                st.warning("Please enable audio generation in settings first.")
    
    with col4:
        if st.button("🔄 Regenerate Text", use_container_width=True):
            st.session_state.audio_bytes = None
            st.rerun()

# Features section
st.markdown("---")
st.markdown("### 🌟 Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class='feature-card'>
            <h3>🤖 AI-Powered</h3>
            <p>Advanced LSTM neural network for accurate predictions</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='feature-card'>
            <h3>⚡ Real-Time</h3>
            <p>Instant word predictions as you type</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class='feature-card'>
            <h3>🎨 Beautiful UI</h3>
            <p>Modern, responsive design for best experience</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class='feature-card'>
            <h3>📊 Analytics</h3>
            <p>Track your usage and generation statistics</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <p>Built with ❤️ using Streamlit & TensorFlow</p>
        <p>Next Word Prediction using LSTM Neural Networks</p>
    </div>
""", unsafe_allow_html=True)
