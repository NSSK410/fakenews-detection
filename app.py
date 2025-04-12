import streamlit as st
import joblib
import re

# Load the model and vectorizer using joblib
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Streamlit UI
st.title("Fake News Classifier")
st.markdown("Enter the news article below to classify it as Real or Fake.")

# Text input field
news_text = st.text_area("News Article:")

# Button to trigger classification
if st.button("Classify"):
    if news_text:
        cleaned_text = clean_text(news_text)
        transformed = vectorizer.transform([cleaned_text])  # Transform the text using vectorizer
        prediction = model.predict(transformed)[0]  # Make the prediction using the model

        # Show the result
        if prediction == 1:
            st.success("This news is likely **Real** ✅")
        else:
            st.error("This news is likely **Fake** ❌")
    else:
        st.warning("Please enter some text to classify.")
