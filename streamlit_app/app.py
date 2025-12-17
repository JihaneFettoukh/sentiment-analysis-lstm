import streamlit as st
import requests

# URL de ton API FastAPI
API_URL = "http://api:8000/predict"

st.set_page_config(page_title="Book Sentiment Analysis", layout="centered")


st.title("📝 Text Sentiment Analysis")
st.write("Enter any text and get its sentiment prediction.")

# Zone de texte
review = st.text_area("✍️ Enter your text here", height=150)


# Bouton de prédiction
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        data = {"text": review}
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()
            sentiment = result["sentiment"]

            if sentiment == "Positive":
                st.success(f"😊 Sentiment: {sentiment}")
            elif sentiment == "Negative":
                st.error(f"😡 Sentiment: {sentiment}")
            else:
                st.info(f"😐 Sentiment: {sentiment}")
        else:
            st.error("Error connecting to the API")
