import streamlit as st
import pdfplumber
from transformers import pipeline
from gtts import gTTS
import os

# Load summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to summarize text
def summarize_text(text):
    # Truncate text if too long
    max_length = 1024
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + " "
    return summary

# Function to convert text to speech
def text_to_speech(text, filename="summary.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# Streamlit UI
st.title("📄 PDF Summarizer with Speech Output")
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file:
    with st.spinner("Extracting text..."):
        raw_text = extract_text_from_pdf(pdf_file)
    with st.spinner("Summarizing..."):
        summary = summarize_text(raw_text)
    st.subheader("📝 Summary:")
    st.write(summary)

    with st.spinner("Converting to speech..."):
        audio_path = text_to_speech(summary)
    st.audio(audio_path)
    st.success("Done!")
