import streamlit as st
import pdfplumber
from transformers import pipeline
from gtts import gTTS
import io

# --- 1. CONFIGURATION AND CACHING ---

# Set Streamlit page configuration for a beautiful, wide layout
st.set_page_config(
    page_title="AI PDF Summarizer & Speaker",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the large summarization model to prevent re-loading on every interaction
# This significantly improves performance.
@st.cache_resource
def load_summarizer():
    # Model used for summarization
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Cache the text extraction operation. It only re-runs if the uploaded file changes.
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_file):
    """Extracts text from a file-like PDF object."""
    text = ""
    try:
        # pdf_file is a file-like object from st.file_uploader.
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error during PDF extraction. The file might be corrupted or image-only: {e}")
        return ""

# --- 2. CORE PROCESSING FUNCTIONS ---

def summarize_text(summarizer, text):
    """Summarizes text by chunking and processing sequentially."""
    MAX_CHUNK_SIZE = 1024
    SUMMARY_MAX_LENGTH = 120
    SUMMARY_MIN_LENGTH = 30

    if not text:
        return "No text extracted for summarization."

    # Chunk the text for the model (BART-large-cnn has a token limit)
    text_length = len(text)
    chunks = [text[i:i + MAX_CHUNK_SIZE] for i in range(0, text_length, MAX_CHUNK_SIZE)]
    summary_parts = []

    # Display progress to the user
    progress_bar = st.progress(0, text="Starting summarization...")

    for i, chunk in enumerate(chunks):
        try:
            result = summarizer(
                chunk,
                max_length=SUMMARY_MAX_LENGTH,
                min_length=SUMMARY_MIN_LENGTH,
                do_sample=False
            )
            summary_parts.append(result[0]['summary_text'])

            # Update progress bar
            percent_complete = int(((i + 1) / len(chunks)) * 100)
            progress_bar.progress(percent_complete, text=f"Summarizing chunk {i+1} of {len(chunks)}...")

        except Exception as e:
            st.warning(f"Could not summarize chunk {i+1}. Skipping. Error: {e}")
            continue

    progress_bar.empty()
    return " ".join(summary_parts)

def text_to_speech(text):
    """Converts text to speech and returns the audio data in an in-memory buffer."""
    mp3_fp = io.BytesIO()
    try:
        # Write the MP3 content directly to the in-memory buffer, avoiding disk operations
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0) # Rewind the buffer to the start for st.audio
        return mp3_fp
    except Exception as e:
        # st.error is called in the main block
        return None

# --- 3. STREAMLIT UI LAYOUT ---

# Sidebar for controls and information
with st.sidebar:
    st.markdown(
        """
        ## 🎙️ AI Reader

        **Turn lengthy PDFs into concise, audible summaries.**

        This application uses:
        - `pdfplumber` for robust text extraction.
        - **BART-large-cnn** for advanced summarization.
        - `gTTS` for natural-sounding speech generation.
        """
    )
    # The main user input
    pdf_file = st.file_uploader("📂 Upload Your PDF Document", type=["pdf"])

    st.markdown("---")
    # Reset button to clear results and process a new file easily
    if st.button("🔄 Process New PDF", help="Clear the current results and start over."):
        for key in list(st.session_state.keys()):
            if key in ['summary', 'audio_data', 'raw_text_length']:
                del st.session_state[key]
        st.rerun()

    st.sidebar.caption("Built with Streamlit and open-source models.")


# Main Application Area
st.title("📄 Intelligent PDF Summary & Reader")
st.markdown("A simple, fast tool to get the TL;DR and audio for any document.")
st.markdown("---")

if pdf_file is None:
    st.info("⬆️ Please upload a PDF file in the sidebar to begin processing.")
else:
    # Use session state to store and retrieve results, preventing re-computation
    if 'summary' not in st.session_state:
        # Load the cached model
        summarizer = load_summarizer()

        # Step 1: Extract Text
        with st.container(border=True):
            st.subheader("Step 1: 📄 Text Extraction")
            with st.spinner("Extracting content from PDF..."):
                raw_text = extract_text_from_pdf(pdf_file)

            if raw_text:
                st.success("Text extracted successfully!")
                st.session_state.raw_text_length = len(raw_text)
                with st.expander(f"Show Extracted Text (Total {len(raw_text):,} characters)"):
                    st.code(raw_text[:1500] + "..." if len(raw_text) > 1500 else raw_text)
            else:
                st.error("Could not extract text. Please ensure the PDF is not an image-only scan.")
                st.stop() # Stop execution if extraction fails


        # Step 2: Summarization and Speech Generation
        st.subheader("Step 2: ✨ Generating Summary & Audio")
        col1, col2 = st.columns(2)

        # --- Summarize ---
        with col1:
            with st.container(border=True):
                st.markdown("#### 🧠 Summary Generation")
                with st.spinner("Summarizing content... This may take a moment."):
                    summary = summarize_text(summarizer, raw_text)
                st.session_state.summary = summary

                st.markdown("<h5 style='color: #2ecc71;'>✅ Summary Result:</h5>", unsafe_allow_html=True)
                st.text_area(
                    "Generated Summary",
                    summary,
                    height=250,
                    help="The summary is generated using the BART-large-cnn model."
                )
                st.metric(label="Original Size", value=f"{st.session_state.raw_text_length:,} characters")


        # --- Text to Speech ---
        with col2:
            with st.container(border=True):
                st.markdown("#### 🎙️ Text-to-Speech")
                with st.spinner("Converting summary to speech..."):
                    audio_data = text_to_speech(summary)
                st.session_state.audio_data = audio_data

                if audio_data:
                    st.markdown("<h5 style='color: #2ecc71;'>✅ Audio Ready:</h5>", unsafe_allow_html=True)
                    st.audio(audio_data, format='audio/mp3')
                    st.success("All processes complete! Read and listen to your summary below.")
                else:
                    st.error("Audio generation failed.")

    # Re-display results if they are already in session state (i.e., user interacted with a non-input widget)
    else:
        st.subheader("Results from Previously Uploaded Document")
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("#### 🧠 Summary Result")
                st.text_area(
                    "Generated Summary",
                    st.session_state.summary,
                    height=300
                )
                st.metric(label="Original Size", value=f"{st.session_state.raw_text_length:,} characters")

        with col2:
            with st.container(border=True):
                st.markdown("#### 🎙️ Audio Playback")
                st.audio(st.session_state.audio_data, format='audio/mp3')
                st.success("Summary and Audio loaded from cache.")