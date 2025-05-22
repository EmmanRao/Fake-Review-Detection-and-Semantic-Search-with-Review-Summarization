import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

# === Custom CSS Styling ===
st.markdown("""
    <style>
        /* Background */
        body, .main, .block-container {
            background-color: white !important;
        }

        /* Text Area Styling */
        textarea {
            background-color: white !important;
            color: black !important;
            border: 1px solid #4B0082 !important;
            border-radius: 8px !important;
        }

        .stTextArea > div > textarea {
            background-color: white !important;
            color: black !important;
        }

        /* Headings */
        h1, h3, p {
            color: #4B0082 !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: white !important;
            color: #4B0082 !important;
            border: 1px solid #4B0082 !important;
            border-radius: 10px !important;
            padding: 0.6em 1.2em;
            font-weight: 600;
            box-shadow: 0px 4px 8px rgba(75, 0, 130, 0.2);
            transition: 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #f0eaff !important;
            box-shadow: 0px 6px 12px rgba(75, 0, 130, 0.3);
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# === App Title ===
st.markdown("<h1 style='text-align:center;'>🕵️‍♀️ Fake Review Detector</h1>", unsafe_allow_html=True)

# === Review Classification ===
st.markdown("<h3>🔍 Check if a Review is Fake or Real</h3>", unsafe_allow_html=True)
st.markdown("<p>✏️ Enter a review to classify:</p>", unsafe_allow_html=True)
review_text = st.text_area("", key="review_classify")

if st.button("🚦 Predict"):
    if review_text:
        response = requests.post(f"{API_URL}/predict-fake-review", json={"review_text": review_text})
        if response.status_code == 200:
            result = response.json()
            st.success(f"📝 Review: {result['review']}")
            st.success(f"🎯 Prediction: {result['prediction']}")
        else:
            st.error("⚠️ Error getting response from backend.")
        

# === Semantic Search ===
st.markdown("<h3>📚 Find Similar Reviews</h3>", unsafe_allow_html=True)
st.markdown("<p>🔎 Enter a review to find similar ones:</p>", unsafe_allow_html=True)
similar_review_text = st.text_area("", key="review_search")

if st.button("📋 Find Similar Reviews"):
    if similar_review_text:
        response = requests.post(f"{API_URL}/semantic-search", json={"review_text": similar_review_text})
        if response.status_code == 200:
            result = response.json()
            st.markdown(f"<p>🧾 <strong>Query:</strong> {result['query']}</p>", unsafe_allow_html=True)
            st.markdown("<p>💡 <strong>Top 5 Similar Reviews:</strong></p>", unsafe_allow_html=True)
            for res in result["results"]:
                st.markdown(f"<p>- 🔹 <strong>Score:</strong> `{res['score']:.4f}` | 📝 {res['review']}</p>", unsafe_allow_html=True)
        else:
            st.error("⚠️ Error getting response from backend.")

# === Explainability ===
st.markdown("<h3>🧠 Summarize the Review</h3>", unsafe_allow_html=True)
st.markdown("<p>🗣️ Summarized Review:</p>", unsafe_allow_html=True)
explanation_review = st.text_area("", key="review_explain")

if st.button("🧾 Explain"):
    if explanation_review:
        response = requests.post(f"{API_URL}/explain-review", json={"review_text": explanation_review})
        if response.status_code == 200:
            result = response.json()
            st.markdown(f"<p>📝 <strong>Review:</strong> {result['review']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>🧠 <strong>Explanation:</strong> {result['explanation']}</p>", unsafe_allow_html=True)
        else:
            st.error("⚠️ Error getting response from backend.")
