# app.py
# Extended from your original app — preserves all original pipelines and structure,
# and adds:
#  - smooth scroll + section highlight
#  - colored wordclouds (keywords colored by sentiment)
#  - PDF report generation (summary, sentiment, emotions chart, keywords & wordcloud)
#
# NOTE: Keep dependencies installed (see comments above)

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
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tempfile
import os
from PyPDF2 import PdfReader
# import whisper
from fpdf import FPDF

# ----------------------------
# Config & Models (unchanged base)
# ----------------------------
st.set_page_config(page_title="Advanced Sentiment & Emotions", layout="wide", page_icon="🤖")

@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    fine_grained = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return sentiment, emotion, summarizer, fine_grained

sentiment_analyzer, emotion_analyzer, summarizer, fine_grained_analyzer = load_models()

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# ----------------------------
# Helpers (original + new)
# ----------------------------
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks, start, L = [], 0, len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if end == L: break
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

# ----------------------------
# Original analysis functions (kept)
# ----------------------------
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    label = mapping.get(result["label"], result["label"])
    return {"label": label, "score": round(result["score"]*100, 2)}

def analyze_fine_grained(text):
    result = fine_grained_analyzer(text)[0]
    label = result["label"]
    score = round(result["score"]*100, 2)
    mapping = {
        "1 star": "Very Negative",
        "2 stars": "Negative",
        "3 stars": "Neutral",
        "4 stars": "Positive",
        "5 stars": "Very Positive"
    }
    return {"label": mapping.get(label, label), "score": score}

def analyze_emotions_aggregate(text):
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    if not chunks:
        return {}
    weighted, total_len = {}, 0
    for ch in chunks:
        out = emotion_analyzer(ch)[0]  # list of dicts
        L = len(ch)
        total_len += L
        for item in out:
            lbl, sc = item["label"], item["score"]
            weighted[lbl] = weighted.get(lbl, 0.0) + sc * L
    for k in list(weighted.keys()):
        weighted[k] = round((weighted[k] / total_len) * 100, 2)
    return dict(sorted(weighted.items(), key=lambda x: x[1], reverse=True))

def summarize_text(text, max_len=150):
    short = text if len(text) <= 1000 else text[:1000]
    try:
        return summarizer(short, max_length=max_len, min_length=30, do_sample=False)[0]["summary_text"]
    except Exception:
        return ""

# ----------------------------
# New helpers: keywords, highlighting, wordcloud (colored), transcription, pdf
# ----------------------------
def extract_keywords(text, top_n=10):
    doc = nlp(text)
    chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
    if not chunks:
        return []
    keywords = pd.Series([c.lower() for c in chunks]).value_counts().head(top_n).index.tolist()
    return keywords

def highlight_keywords(text, keywords):
    if not keywords:
        return text
    # longer keywords first to avoid partial overlaps
    keywords_sorted = sorted(set(keywords), key=lambda x: -len(x))
    escaped = [re.escape(k) for k in keywords_sorted]
    pattern = r"(?i)\b(" + "|".join(escaped) + r")\b"
    # wrap with mark and add an id to the first match for focus (optional)
    return re.sub(pattern, r"<mark style='background:yellow'>\1</mark>", text)

def generate_wordcloud_from_freq(freq_dict, color_map=None):
    # color_map: optional dict mapping word -> color
    wc = WordCloud(width=800, height=400, background_color="white")
    # generate mask-less
    # if custom coloring is desired, define color_func
    if color_map:
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return color_map.get(word.lower(), "rgb(0,0,0)")
        wc = WordCloud(width=800, height=400, background_color="white", color_func=color_func)
        img = wc.generate_from_frequencies(freq_dict)
    else:
        img = wc.generate_from_frequencies(freq_dict)
    fig_wc, ax = plt.subplots(figsize=(10,5))
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")
    return fig_wc

# def transcribe_audio(file):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
#             tmp.write(file.read())
#             tmp.flush()
#             tmp_path = tmp.name
#         result = whisper_model.transcribe(tmp_path)
#         try:
#             os.remove(tmp_path)
#         except Exception:
#             pass
#         return result.get("text", "")
#     except Exception as e:
#         st.error(f"Transcription failed: {e}")
#         return ""

def save_plotly_fig_as_png(fig):
    """Try to convert plotly figure to PNG using kaleido if available; else return None."""
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png")
    except Exception:
        return None

def generate_pdf_report(summary_text, basic_sent, fine_sent, df_emotions, fig_emotion_matplotlib, df_kw, fig_kw_matplotlib):
    """
    Create a PDF report (FPDF) and return bytes.
    We embed summary, sentiment text, emotion table, and images (charts/wordcloud) saved as temp PNGs.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 8, "Sentiment & Emotion Analysis Report", ln=True, align="C")

    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 6, f"Summary:\n{summary_text}\n")

    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 6, f"Basic Sentiment: {basic_sent['label']} ({basic_sent['score']}%)")
    pdf.multi_cell(0, 6, f"Fine-Grained Sentiment: {fine_sent['label']} ({fine_sent['score']}%)")

    # Emotions table
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 6, "Emotions:", ln=True)
    if df_emotions is not None and not df_emotions.empty:
        # write table rows
        pdf.set_font("Arial", size=10)
        for i, row in df_emotions.iterrows():
            pdf.cell(0, 5, f"{row['Emotion']}: {row['Score(%)']}%", ln=True)
    else:
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, "No emotion data available.", ln=True)

    # Save matplotlib figs to temp PNGs and add to PDF
    temp_files = []
    try:
        if fig_emotion_matplotlib:
            tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_emotion_matplotlib.savefig(tmp1.name, bbox_inches="tight")
            temp_files.append(tmp1.name)
            pdf.add_page()
            pdf.image(tmp1.name, x=15, y=20, w=180)
        if fig_kw_matplotlib:
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_kw_matplotlib.savefig(tmp2.name, bbox_inches="tight")
            temp_files.append(tmp2.name)
            pdf.add_page()
            pdf.image(tmp2.name, x=15, y=20, w=180)
    except Exception as e:
        # if image insertion fails, continue
        pass

    # Keywords table
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 6, "Keywords:", ln=True)
    if df_kw is not None and not df_kw.empty:
        pdf.set_font("Arial", size=10)
        for i, row in df_kw.iterrows():
            pdf.multi_cell(0, 5, f"{row['Keyword']}  —  {row['Sentiment']} ({row['Confidence']}%)")
    else:
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, "No keywords available.", ln=True)

    # collect bytes
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(output_path)
    with open(output_path, "rb") as f:
        data = f.read()
    # cleanup temp files
    try:
        os.remove(output_path)
        for p in temp_files:
            os.remove(p)
    except Exception:
        pass
    return data

# ----------------------------
# UI Input (preserve original)
# ----------------------------
st.title("🤖 Summarizer + Sentiment & Emotions Analyzer")
st.write("Paste text, upload a file, provide a URL, or upload audio/video to analyze. All results display on a single page; use the navigation bar to jump to sections.")

# mode = st.radio("Input method:", ["✍️ Text", "🌐 URL", "📂 File", "🎤 Audio/Video"], horizontal=True)
mode = st.radio("Input method:", ["✍️ Text", "🌐 URL", "📂 File"], horizontal=True)

input_text = ""
if mode == "✍️ Text":
    input_text = st.text_area("Paste or type text:", height=260, value=st.session_state.get("input_text",""))
    st.session_state["input_text"] = input_text

elif mode == "🌐 URL":
    url = st.text_input("Enter URL:")
    if st.button("Fetch URL"):
        fetched = fetch_url_text(url)
        if fetched:
            input_text = (st.session_state.get("input_text","") + "\n\n" + fetched).strip()
            st.session_state["input_text"] = input_text
            st.success("Fetched and appended.")
        else:
            st.warning("No text extracted from URL.")

elif mode == "📂 File":
    uploaded = st.file_uploader("Upload TXT or PDF", type=["txt","pdf"])
    if uploaded:
        if uploaded.type == "text/plain":
            input_text = uploaded.read().decode("utf-8", errors="ignore")
        elif uploaded.type == "application/pdf":
            try:
                reader = PdfReader(uploaded)
                pages = [p.extract_text() or "" for p in reader.pages]
                input_text = "\n\n".join(pages)
            except Exception as e:
                st.error("Failed to read PDF: " + str(e))
                input_text = ""
        st.session_state["input_text"] = (st.session_state.get("input_text","") + "\n\n" + input_text).strip()
        st.success("File appended.")

# elif mode == "🎤 Audio/Video":
#     audio_file = st.file_uploader("Upload audio/video file (mp3,wav,m4a,mp4)", type=["mp3","wav","m4a","mp4"])
#     if audio_file:
#         st.info("Transcribing with Whisper... (may take a while on first run)")
#         transcribed = transcribe_audio(audio_file)
#         if transcribed:
#             input_text = (st.session_state.get("input_text","") + "\n\n" + transcribed).strip()
#             st.session_state["input_text"] = input_text
#             st.success("Audio transcription complete.")
#         else:
#             st.warning("No text transcribed.")

working_text = st.session_state.get("input_text","").strip()

# preserve buttons
c1, c2, c3 = st.columns([1,1,1])
with c1: do_analyze = st.button("🔍 Analyze")
with c2: do_summarize = st.button("📝 Summarize")
with c3: do_both = st.button("⚡ Both")

# navigation radio (acts as tabs) — user chooses where to jump
nav_choice = st.radio("📍 Jump to Section:", ["Original Text", "Summary", "Sentiment", "Emotions", "Keywords"], horizontal=True)

# JS helpers for smooth scroll + highlight
def render_scroll_and_flash(target_id):
    if not target_id:
        return
    # JS: scroll into view, add a flashing border/background, remove after 1.6s
    script = f"""
    <script>
    (function() {{
        var el = document.getElementById("{target_id}");
        if (el) {{
            el.scrollIntoView({{behavior: "smooth", block: "start"}});
            // add highlight
            var orig = el.style.boxShadow;
            el.style.transition = "box-shadow 0.3s, background-color 0.3s";
            el.style.boxShadow = "0 0 0 4px rgba(255,200,0,0.6)";
            el.style.backgroundColor = "rgba(255,250,200,0.4)";
            setTimeout(function() {{
                el.style.boxShadow = orig;
                el.style.backgroundColor = "";
            }}, 1600);
        }}
    }})();
    </script>
    """
    components.html(script, height=0)

# main run
if (do_analyze or do_summarize or do_both):
    if not working_text:
        st.warning("Please provide text first.")
    else:
        text = clean_text(working_text)

        # Precompute outputs
        result_summary = None
        if do_summarize or do_both:
            with st.spinner("Summarizing..."):
                result_summary = summarize_text(text, max_len=160)

        if do_analyze or do_both:
            base_text = text if not do_both else (result_summary or text)
            with st.spinner("Analyzing polarity & emotions..."):
                basic = analyze_sentiment(base_text)
                fine = analyze_fine_grained(base_text)
                emotions = analyze_emotions_aggregate(base_text)

        # Keywords & highlighted original
        keywords = extract_keywords(text, top_n=10)
        highlighted_text = highlight_keywords(text, keywords)

        # Prepare emotions df
        df_emotions = None
        if emotions:
            df_emotions = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Score(%)"]).sort_values("Score(%)", ascending=False)

        # Render Single Page sections with anchors
        # Original
        st.markdown(f"<h2 id='section-original'>📜 Original Text with Highlights</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding:10px;border:1px solid #ddd;border-radius:6px' id='orig-box'>{highlighted_text}</div>", unsafe_allow_html=True)
        copy_button(text, "Copy Original")

        # Summary
        st.markdown(f"<h2 id='section-summary'>📝 Summary</h2>", unsafe_allow_html=True)
        if result_summary:
            st.write(result_summary)
            copy_button(result_summary or "", "Copy Summary")
        else:
            st.info("Summary not generated. Click 'Summarize' or 'Both' to generate a summary.")

        # Sentiment
        st.markdown(f"<h2 id='section-sentiment'>🌓 Sentiment Polarity</h2>", unsafe_allow_html=True)
        if do_analyze or do_both:
            st.write(f"**Basic Polarity:** {basic['label']} ({basic['score']}%)")
            st.write(f"**Fine-Grained Polarity:** {fine['label']} ({fine['score']}%)")
        else:
            st.info("Polarity not generated. Click 'Analyze' or 'Both' to run sentiment analysis.")

        # Emotions
        st.markdown(f"<h2 id='section-emotions'>🎭 Emotion Distribution</h2>", unsafe_allow_html=True)
        if df_emotions is not None and not df_emotions.empty:
            st.dataframe(df_emotions, use_container_width=True)

            # # Matplotlib bar chart (for PDF embedding and for display)
            # fig_mat, ax = plt.subplots(figsize=(8, 4))
            # ax.bar(df_emotions["Emotion"], df_emotions["Score(%)"], color=px.colors.qualitative.Plotly)
            # ax.set_ylabel("Score (%)")
            # ax.set_title("Emotion Distribution")
            # plt.xticks(rotation=45)
            # st.pyplot(fig_mat)

            # Plotly chart (for interactive display)
            df_chart = df_emotions.sort_values("Score(%)", ascending=True)
            fig = px.bar(df_chart, x="Score(%)", y="Emotion", orientation="h",
                         color="Score(%)", color_continuous_scale="RdYlBu",
                         text="Score(%)", title="Emotion Distribution")
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(xaxis_title="Score (%)", yaxis_title="", template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Download interaction as PNG or HTML fallback
            img_bytes = save_plotly_fig_as_png(fig)
            if img_bytes:
                st.download_button("⬇️ Download Emotion Chart (PNG)", data=img_bytes, file_name="emotion_chart.png", mime="image/png")
            else:
                html_bytes = fig.to_html().encode("utf-8")
                st.download_button("⬇️ Download Emotion Chart (HTML)", data=html_bytes, file_name="emotion_chart.html", mime="text/html")

            # Emotion wordcloud (colored by emotion importance)
            color_map_emotion = {}
            # map emotion label -> color (simple palette)
            palette = {
                "joy": "rgb(255,165,0)",      # orange
                "fear": "rgb(128,0,128)",     # purple
                "sadness": "rgb(30,144,255)", # dodgerblue
                "anger": "rgb(220,20,60)",    # crimson
                "surprise": "rgb(34,139,34)"  # forestgreen
            }
            # generate freq dict from emotions for the wordcloud
            freq_em = {k: max(1, int(v)) for k, v in emotions.items()}
            # map names to colors lowercased
            for k in freq_em.keys():
                color_map_emotion[k.lower()] = palette.get(k.lower(), "rgb(0,0,0)")
            st.subheader("☁️ Emotion Wordcloud")
            fig_wc_em = generate_wordcloud_from_freq(freq_em, color_map=color_map_emotion)
            st.pyplot(fig_wc_em)
        else:
            st.info("Emotion data not available. Click 'Analyze' or 'Both' to compute emotions.")
            fig_mat = None
            fig_wc_em = None

        # Keywords
        st.markdown(f"<h2 id='section-keywords'>🔑 Keywords & Sentiment</h2>", unsafe_allow_html=True)
        if keywords:
            results = []
            for kw in keywords:
                sent = analyze_sentiment(kw)
                results.append({"Keyword": kw, "Sentiment": sent["label"], "Confidence": sent["score"]})
            df_kw = pd.DataFrame(results)
            st.dataframe(df_kw, use_container_width=True)

            # Wordcloud for keywords colored by sentiment (positive green, neutral gray, negative red)
            color_map_kw = {}
            sentiment_color = {"Positive": "rgb(34,139,34)", "Neutral": "rgb(128,128,128)", "Negative": "rgb(220,20,60)"}
            freq_map = {}
            for row in results:
                k = row["Keyword"].lower()
                color_map_kw[k] = sentiment_color.get(row["Sentiment"], "rgb(0,0,0)")
                # weight approx: confidence (0-100) -> bucket
                freq_map[k] = max(1, int(row["Confidence"] // 10))
            st.subheader("🔍 Keyword Wordcloud (colored by sentiment)")
            fig_kw = generate_wordcloud_from_freq(freq_map, color_map=color_map_kw)
            st.pyplot(fig_kw)
        else:
            df_kw = pd.DataFrame()
            fig_kw = None
            st.info("No keywords extracted. Provide more text or click 'Analyze'/'Both'.")

        # Downloadable PDF report
        st.markdown("### 📄 PDF Report")
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                # Use matplotlib figures we created: fig_mat (emotion bar), fig_kw (keyword wc)
                # If fig_mat/fig_kw None, try to build minimal images
                try:
                    # pass df_emotions, fig_mat, df_kw, fig_kw to PDF generator
                    pdf_bytes = generate_pdf_report(result_summary or "", basic if 'basic' in locals() else {"label":"N/A","score":0},
                                                    fine if 'fine' in locals() else {"label":"N/A","score":0},
                                                    df_emotions, fig_mat if 'fig_mat' in locals() else None,
                                                    df_kw, fig_kw)
                    st.download_button("⬇️ Download PDF Report", data=pdf_bytes, file_name="analysis_report.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        # After rendering, scroll+flash to selected nav_choice
        id_map = {
            "Original Text": "section-original",
            "Summary": "section-summary",
            "Sentiment": "section-sentiment",
            "Emotions": "section-emotions",
            "Keywords": "section-keywords"
        }
        target = id_map.get(nav_choice)
        render_scroll_and_flash(target)

st.markdown("---")
st.markdown("Notes: first run may be slower while transformer models download. Input under a few thousand characters recommended for best interactive speed.")
