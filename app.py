# app.py
import streamlit as st
from transformers import pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import json
import uuid

# ---------------------------
# Config / Limits
# ---------------------------
st.set_page_config(page_title="Advanced Sentiment + Summarizer", page_icon="🤖", layout="wide")

# Recommended character limits (you can change these)
CHAR_LIMIT = 4000          # max characters allowed in the input box (UI-level)
CHUNK_SIZE = 1000          # chunk size for model inputs (characters)
CHUNK_OVERLAP = 200        # overlap between chunks to preserve context

st.title("🤖 Advanced Emotion + Summarizer Toolkit")
st.write(
    "Paste text below (or upload a file). App will run emotion analysis (many emotions) "
    "and produce a smart summary. Large inputs are chunked automatically."
)

# ---------------------------
# Helpers: chunking, copy button
# ---------------------------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks (by characters)"""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        # avoid infinite loop
        if end == L:
            break
    return chunks

def copy_button(text: str, label: str = "Copy"):
    """Render a small copy-to-clipboard button for `text` using an iframe component."""
    if not text:
        st.write("Nothing to copy.")
        return
    btn_id = "btn_" + uuid.uuid4().hex
    js_text = json.dumps(text)  # safe JS string literal
    html = f"""
    <button id="{btn_id}">{label}</button>
    <script>
    const btn = document.getElementById("{btn_id}");
    btn.addEventListener('click', () => {{
        navigator.clipboard.writeText({js_text}).then(() => {{
            btn.innerText = 'Copied!';
        }}).catch(err => {{
            alert('Copy failed: ' + err);
        }});
    }});
    </script>
    """
    components.html(html, height=40)

# ---------------------------
# Load models (cached)
# ---------------------------
@st.cache_resource
def load_summarizer():
    # BART summarizer
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_emotion_model():
    # Multi-emotion model (returns all scores)
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

summarizer = load_summarizer()
emotion_model = load_emotion_model()

# ---------------------------
# Input area + controls
# ---------------------------
st.subheader("Input Text")
st.info(f"Max characters allowed: {CHAR_LIMIT} (app will chunk long text for model APIs).")

# Use session_state-backed text_area so input persists across reruns
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

text = st.text_area(
    "✍️ Paste or type text here:",
    value=st.session_state["input_text"],
    height=260,
    max_chars=CHAR_LIMIT,
    key="input_text"
)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    do_analyze = st.button("🔍 Analyze Emotions")
with col_b:
    do_summarize = st.button("📝 Summarize Text")
with col_c:
    do_both = st.button("⚡ Analyze & Summarize")

# ---------------------------
# Utility: emotion analysis with chunking
# ---------------------------
def analyze_emotions_aggregate(text: str):
    """Analyze text in chunks and produce aggregated emotion scores (percentage)."""
    chunks = chunk_text(text)
    if not chunks:
        return {}
    # accumulate weighted scores by chunk length
    weight_scores = {}
    total_len = 0
    for chunk in chunks:
        # each call returns a list containing the list-of-scores: e.g. [ [ {label,score}, ... ] ]
        model_out = emotion_model(chunk)[0]  # list of dicts
        L = len(chunk)
        total_len += L
        for item in model_out:
            lbl = item["label"]
            sc = item["score"]
            weight_scores[lbl] = weight_scores.get(lbl, 0.0) + sc * L

    # normalize to percentages
    for k in weight_scores.keys():
        weight_scores[k] = round((weight_scores[k] / total_len) * 100, 2)

    # sort descending
    sorted_scores = dict(sorted(weight_scores.items(), key=lambda x: x[1], reverse=True))
    return sorted_scores

# ---------------------------
# Utility: summarization with chunking
# ---------------------------
def summarize_long_text(text: str):
    """Chunk long text and summarize each chunk, then (optionally) summarize the concatenation."""
    chunks = chunk_text(text)
    if not chunks:
        return ""
    chunk_summaries = []
    for ch in chunks:
        # tune max_length/min_length as you like
        try:
            out = summarizer(ch, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        except Exception:
            # fallback: try a safer shorter summary
            out = summarizer(ch, max_length=100, min_length=20, do_sample=False)[0]["summary_text"]
        chunk_summaries.append(out)

    # if multiple chunk summaries, summarize them again to produce a coherent final summary
    if len(chunk_summaries) > 1:
        joined = " ".join(chunk_summaries)
        final = summarizer(joined, max_length=180, min_length=40, do_sample=False)[0]["summary_text"]
        return final
    else:
        return chunk_summaries[0]

# ---------------------------
# Run requested actions
# ---------------------------
if do_analyze or do_summarize or do_both:
    if not text.strip():
        st.warning("Please enter some text to analyze or summarize.")
    else:
        # Show original text preview + copy button
        st.markdown("### Original Text (preview)")
        st.write(text[:4000] + ("..." if len(text) > 4000 else ""))
        copy_button(text, label="Copy Original Text")

        if do_analyze or do_both:
            with st.spinner("Analyzing emotions..."):
                emotion_scores = analyze_emotions_aggregate(text)
            if not emotion_scores:
                st.warning("No emotion scores were returned.")
            else:
                st.subheader("📊 Emotion Scores (aggregated)")
                df_em = pd.DataFrame(list(emotion_scores.items()), columns=["Emotion", "Score(%)"])
                df_em = df_em.sort_values("Score(%)", ascending=False).reset_index(drop=True)
                st.dataframe(df_em, use_container_width=True)
                st.bar_chart(df_em.set_index("Emotion"))  # quick bar chart
                # Show dominant emotion
                top_em = df_em.iloc[0]
                st.markdown(f"### 🎯 Dominant Emotion: **{top_em['Emotion']}** — {top_em['Score(%)']}%")
                copy_button(df_em.to_csv(index=False), label="Copy emotion table (CSV)")

        if do_summarize or do_both:
            with st.spinner("Generating summary..."):
                summary_text = summarize_long_text(text)
            st.subheader("📝 Summary")
            st.write(summary_text)
            copy_button(summary_text, label="Copy Summary")

# ---------------------------
# Helpful troubleshooting text for "can't copy / paste" issues
# ---------------------------
# st.markdown("---")
# st.subheader("Troubleshooting: Can't copy/paste into the text area?")
# st.markdown(
#     """
# Common reasons why selecting or pasting into Streamlit's text area may behave unexpectedly — and how to fix them:
# 1. **`disabled=True` or code clearing the text** — If your app or session code is setting `disabled=True` or explicitly clearing `st.session_state['input_text']` on every run, the box may appear un-editable.
#    **Fix:** Don't set `disabled=True` and avoid clearing `st.session_state` unless the user clicks a clear button.
#
# 2. **Browser extension / adblock / privacy extension** — Some browser extensions interfere with clipboard or contenteditable areas.
#    **Fix:** Try disabling extensions or use another browser (Chrome/Edge/Firefox) to test.
#
# 3. **Streamlit version bug** — Very old versions of Streamlit may have bugs.
#    **Fix:** Upgrade with `pip install --upgrade streamlit`.
#
# 4. **Rapid reruns clearing input** — Button clicks cause a rerun; if your code resets session_state on every run, the text will vanish.
#    **Fix:** Use `key` on `st.text_area` (we do: `key='input_text'`) and avoid resetting that session_state value on every run.
#
# 5. **Restricted clipboard APIs in some environments** — Some corporate setups block `navigator.clipboard`.
#    **Fix:** Use the manual selection + Ctrl+C or try the provided copy buttons (they use the browser clipboard API).
#
# If the above don't fix it, try: open DevTools (F12), check the console for errors, or paste here the error/behavior and I’ll help debug.
# """
# )
#
# st.markdown("**Recommended character limits:** For smooth UX, keep inputs under ~2000 characters. This app supports up to "
#             f"{CHAR_LIMIT} characters and will chunk automatically for the models (BART summarizer and DistilRoBERTa-based emotion model).")
