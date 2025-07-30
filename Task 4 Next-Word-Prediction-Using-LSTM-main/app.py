import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import time
from gtts import gTTS
import io

st.set_page_config(page_title="StoryWeaver", page_icon="🖋️", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #E0F7FA;
        color: #01579B;
        font-family: 'Times New Roman', serif;
    }
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: #01579B;
        border: 1px solid #4FC3F7;
    }
    .stNumberInput > div > div > input {
        background-color: #FFFFFF;
        color: #01579B;
        border: 1px solid #4FC3F7;
    }
    .stButton > button {
        background-color: #4FC3F7;
        color: #FFFFFF;
        border: 1px solid #0288D1;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #0288D1;
        color: #FFFFFF;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #0288D1;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #4FC3F7;
        text-align: center;
        margin-bottom: 30px;
    }
    .output {
        font-size: 20px;
        color: #01579B;
        background-color: #B3E5FC;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4FC3F7;
        margin-top: 20px;
        white-space: pre-wrap;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = load_model("my_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'reset' not in st.session_state:
    st.session_state.reset = False

st.markdown('<div class="title">StoryWeaver</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Craft tales with AI-powered word predictions</div>', unsafe_allow_html=True)

with st.form(key="input_form"):
    seed_text = st.text_input("Enter your starting phrase:", value="When it grew")
    num_words = st.number_input("Number of words to generate:", min_value=1, max_value=50, value=20)
    generate_button = st.form_submit_button(label="Generate Story")
    reset_button = st.form_submit_button(label="Reset")

if reset_button:
    st.session_state.generated_text = ""
    st.session_state.audio_bytes = None
    st.session_state.reset = True
    st.rerun()

def generate_text(seed_text, num_words, model, tokenizer, max_len=17):
    text = seed_text
    output_placeholder = st.empty()
    output_placeholder.markdown('<div class="output">{}</div>'.format(text), unsafe_allow_html=True)
    
    for i in range(num_words):
        tokenize_text = tokenizer.texts_to_sequences([text])[0]
        padded_input_text = pad_sequences([tokenize_text], maxlen=max_len-1, padding='pre')
        pos = np.argmax(model.predict(padded_input_text, verbose=0))
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                output_placeholder.markdown('<div class="output">{}</div>'.format(text), unsafe_allow_html=True)
                time.sleep(0.3)
                break
    st.session_state.generated_text = text
    
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    st.session_state.audio_bytes = audio_fp

if generate_button and seed_text:
    st.session_state.generated_text = seed_text
    generate_text(seed_text, num_words, model, tokenizer)
elif generate_button and not seed_text:
    st.error("Please enter a starting phrase.")

if st.session_state.audio_bytes:
    if st.button("Play Audio"):
        st.audio(st.session_state.audio_bytes, format='audio/mp3')

if st.session_state.generated_text and not st.session_state.reset:
    st.markdown('<div class="output">{}</div>'.format(st.session_state.generated_text), unsafe_allow_html=True)