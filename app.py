# streamlit_nlp_app/app.py

import streamlit as st
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter









# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Page title
st.title("🧠 NLP Player Feedback Analyzer")
st.markdown("Analyze player feedback with token cleaning, sentiment analysis, and named entity recognition.")

# Text input
user_input = st.text_area("Paste player feedback here:", height=150)

if st.button("Analyze") and user_input:
    doc = nlp(user_input)

    # Token Cleaning
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    st.subheader("🧹 Cleaned Tokens")
    st.write(tokens)

    # Named Entities
    st.subheader("🏷 Named Entities")
    if doc.ents:
        for ent in doc.ents:
            st.markdown(f"**{ent.text}** — {ent.label_}")
    else:
        st.write("No named entities found.")

    # Sentiment Analysis
    blob = TextBlob(user_input)
    polarity = round(blob.sentiment.polarity, 3)
    sentiment_label = (
        "positive" if polarity > 0.1 else
        "negative" if polarity < -0.1 else
        "neutral"
    )
    st.subheader("📊 Sentiment")
    st.write(f"**Score:** {polarity}")
    st.write(f"**Label:** {sentiment_label}")

    # Optional: Token Frequency Plot
    st.subheader("🔤 Token Frequency")
    token_counts = Counter(tokens)
    common_tokens = token_counts.most_common(10)
    if common_tokens:
        words, freqs = zip(*common_tokens)
        fig, ax = plt.subplots()
        ax.bar(words, freqs, color="skyblue")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Not enough tokens to generate chart.")
