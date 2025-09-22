import streamlit as st
from transformers import pipeline

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Toolkit",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Toolkit")
st.write("Choose a tool below to summarize text or analyze sentiment.")

# ---------------------------
# Load Models (cached)
# ---------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment()

# ---------------------------
# Tabs for Tools
# ---------------------------
tab1, tab2 = st.tabs(["📝 Text Summarizer", "😊 Sentiment Analyzer"])

# ---------------------------
# Summarizer Tab
# ---------------------------
with tab1:
    st.subheader("📝 Text Summarizer")
    text_input = st.text_area("✍️ Paste your text here:", height=200, key="summarizer")

    if st.button("Summarize", key="sum_btn"):
        if text_input.strip():
            with st.spinner("Summarizing... ⏳"):
                summary = summarizer(
                    text_input,
                    max_length=100,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
            st.success("✅ Summary generated!")
            st.write("### 📌 Summary")
            st.write(summary)
        else:
            st.warning("⚠️ Please enter some text first.")

# ---------------------------
# Sentiment Tab
# ---------------------------
with tab2:
    st.subheader("😊 Sentiment Analyzer")
    user_text = st.text_area("✍️ Type text to analyze sentiment:", height=150, key="sentiment")

    if st.button("Analyze Sentiment", key="sent_btn"):
        if user_text.strip():
            with st.spinner("Analyzing sentiment... ⏳"):
                result = sentiment_analyzer(user_text)[0]
                label = result['label']
                score = round(result['score'] * 100, 2)

            st.success("✅ Analysis complete!")
            st.write(f"**Sentiment:** {label}")
            st.write(f"**Confidence:** {score}%")

            # Emoji feedback
            if label == "POSITIVE":
                st.markdown("😊 This looks **Positive!**")
            elif label == "NEGATIVE":
                st.markdown("😟 This looks **Negative.**")
            else:
                st.markdown("😐 This seems **Neutral.**")
        else:
            st.warning("⚠️ Please enter some text first.")
