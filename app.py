# app.py
import streamlit as st
from transformers import pipeline
import plotly.express as px
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import streamlit.components.v1 as components
import json, uuid
from io import BytesIO

# ----------------------------
# Config & Models
# ----------------------------
st.set_page_config(page_title="Advanced Sentiment & Emotions", layout="wide", page_icon="🤖")

# load models (will download the first time; cache_resource keeps them in memory)
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return sentiment, emotion, summarizer

sentiment_analyzer, emotion_analyzer, summarizer = load_models()

# ----------------------------
# Helpers
# ----------------------------
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=1000, overlap=200):
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
        if end == L:
            break
    return chunks

def copy_button(text: str, label: str = "Copy"):
    if not text:
        return
    btn_id = "btn_" + uuid.uuid4().hex
    safe_text = json.dumps(text)
    html = f"""
    <button id="{btn_id}" style="padding:6px 10px;border-radius:6px;border:1px solid #ccc;background:#f7f7f7;cursor:pointer;">{label}</button>
    <script>
    const btn = document.getElementById("{btn_id}");
    btn.addEventListener('click', () => {{
        navigator.clipboard.writeText({safe_text}).then(() => {{
            btn.innerText = 'Copied!';
            setTimeout(() => btn.innerText = '{label}', 1200);
        }}).catch(err => {{
            alert('Copy failed: ' + err);
        }});
    }});
    </script>
    """
    components.html(html, height=40)

def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        paras = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
        return "\n\n".join(paras)
    except Exception as e:
        st.error(f"URL fetch failed: {e}")
        return ""

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]

def analyze_emotions_aggregate(text):
    """If text is long, chunk and aggregate emotion scores weighted by chunk length."""
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    if not chunks:
        return {}
    weighted = {}
    total_len = 0
    for ch in chunks:
        out = emotion_analyzer(ch)[0]  # list of dicts
        L = len(ch)
        total_len += L
        for item in out:
            lbl = item["label"]
            sc = item["score"]
            weighted[lbl] = weighted.get(lbl, 0.0) + sc * L
    for k in list(weighted.keys()):
        weighted[k] = round((weighted[k] / total_len) * 100, 2)
    # return sorted descending
    return dict(sorted(weighted.items(), key=lambda x: x[1], reverse=True))

def summarize_text(text, max_len=150):
    # summarizer can choke on very long text; send first ~1000 chars to keep it safe
    short = text if len(text) <= 1000 else text[:1000]
    try:
        return summarizer(short, max_length=max_len, min_length=30, do_sample=False)[0]["summary_text"]
    except Exception:
        return ""

# ----------------------------
# UI: Input
# ----------------------------
st.title("🤖 Summarizer and Sentiment Analyzer")
st.write("Paste text, upload a TXT or PDF, or provide a URL. Click Analyze to get sentiment, emotions (table + chart), and a summary.")

mode = st.radio("Input method:", ["✍️ Text", "🌐 URL", "📂 File"], horizontal=True)
# mode = st.radio("Input method:", ["✍️ Text", "🌐 URL"], horizontal=True)


input_text = ""

if mode == "✍️ Text":
    input_text = st.text_area("Paste or type text (no hard limit):", height=260, value=st.session_state.get("input_text",""))
    st.session_state["input_text"] = input_text  # persist

elif mode == "🌐 URL":
    url = st.text_input("Enter URL:")
    if st.button("Fetch URL"):
        fetched = fetch_url_text(url)
        if fetched:
            input_text = (st.session_state.get("input_text","") + "\n\n" + fetched).strip()
            st.session_state["input_text"] = input_text
            st.success("Fetched and appended to text area.")
        else:
            st.warning("No text extracted from URL.")

elif mode == "📂 File":
    uploaded = st.file_uploader("Upload TXT or PDF", type=["txt","pdf"])
    if uploaded:
        if uploaded.type == "text/plain":
            input_text = uploaded.read().decode("utf-8", errors="ignore")
            st.session_state["input_text"] = (st.session_state.get("input_text","") + "\n\n" + input_text).strip()
            st.success("TXT appended to text area.")
        elif uploaded.type == "application/pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(uploaded)
                pages = [p.extract_text() or "" for p in reader.pages]
                full = "\n\n".join(pages)
                input_text = full
                st.session_state["input_text"] = (st.session_state.get("input_text","") + "\n\n" + input_text).strip()
                st.success("PDF text appended to text area.")
            except Exception as e:
                st.error("Failed to read PDF: " + str(e))

# working text
working_text = st.session_state.get("input_text","").strip()

# Action buttons
c1, c2, c3 = st.columns([1,1,1])
with c1: do_analyze = st.button("🔍 Analyze")
with c2: do_summarize = st.button("📝 Summarize")
with c3: do_both = st.button("⚡ Both")

# ----------------------------
# Run analysis & display table + chart
# ----------------------------
if do_analyze or do_summarize or do_both:
    if not working_text:
        st.warning("Please provide text via paste, upload, or URL.")
    else:
        text = clean_text(working_text)
        # show preview and copy
        st.markdown("### Original (preview)")
        st.write(text[:1500] + ("..." if len(text) > 1500 else ""))
        copy_button(text, "Copy Original")

        # summary
        result_summary = None
        if do_summarize or do_both:
            with st.spinner("Summarizing..."):
                result_summary = summarize_text(text, max_len=160)
            st.subheader("📝 Summary")
            st.write(result_summary)
            copy_button(result_summary or "", "Copy Summary")

        # choose whether to analyze original or summary (if both)
        if do_analyze or do_both:
            if do_both:
                choice = st.radio("Which text to analyze for emotions?", ["Original", "Summary"], horizontal=True)
                base_text = text if choice == "Original" else (result_summary or text)
            else:
                base_text = text

            with st.spinner("Analyzing emotions..."):
                emotions = analyze_emotions_aggregate(base_text)

            if not emotions:
                st.warning("Emotion analysis returned no results.")
            else:
                # build DataFrame (descending)
                df = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Score(%)"]).sort_values("Score(%)", ascending=False).reset_index(drop=True)
                st.subheader("🎭 Emotion Table")
                st.dataframe(df, use_container_width=True)   # explicit table display

                # copy and download CSV
                csv_data = df.to_csv(index=False)
                copy_button(csv_data, "Copy Emotions CSV")
                st.download_button("⬇️ Download Emotions CSV", data=csv_data.encode("utf-8"), file_name="emotions.csv", mime="text/csv")

                # horizontal chart: sort ascending for better horizontal ordering (largest on top)
                df_chart = df.sort_values("Score(%)", ascending=True)
                fig = px.bar(
                    df_chart,
                    x="Score(%)",
                    y="Emotion",
                    orientation="h",
                    color="Score(%)",
                    color_continuous_scale="RdYlBu",
                    text="Score(%)",
                    title="Emotion Distribution"
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(xaxis_title="Score (%)", yaxis_title="", template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)

                # allow chart PNG download
                img_bytes = fig.to_image(format="png")
                st.download_button("⬇️ Download Emotion Chart (PNG)", data=img_bytes, file_name="emotion_chart.png", mime="image/png")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "Notes: first run may be slower while transformer models download. Keep input under a few thousand characters for best speed. "
    "If you want larger-file support or PDF/DOCX reports, we can add those while balancing dependencies."
)
