from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# =========================
# Load model & tokenizer
# =========================
model= load_model("lstm_sentiment_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 400  

# =========================
# Stopwords 
# =========================
def get_revised_stopwords():
    from revised_stopwords import get_revised_stopwords
    sw = set(get_revised_stopwords())
    return sw

stopwords_list = get_revised_stopwords()
stopwords_list.discard("but") 

# =========================
# Text preprocessing
# =========================
punc = str.maketrans("", "", string.punctuation)
lem = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # 1. Remove punctuation
    text = " ".join(word.translate(punc) for word in str(text).split())

    # 2. Keep only alphabetical words
    text = " ".join(word for word in text.split() if word.isalpha())

    # 3. Lowercase
    text = text.lower()

    # 4. Remove stopwords
    text = " ".join(word for word in text.split() if word not in stopwords_list)

    # 5. Lemmatization (verbs)
    text = " ".join(lem.lemmatize(word, pos="v") for word in text.split())

    return text

def preprocess_for_model(text: str):
    cleaned = clean_text(text)
    sequences = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    return padded

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Book Review Sentiment Analysis API")

class Review(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running 🚀"}

@app.post("/predict")
def predict_sentiment(review: Review):
    try:
        processed_text = preprocess_for_model(review.text)

        prediction = model.predict(processed_text)

        # Model outputs 0 / 1 / 2
        predicted_label = int(np.argmax(prediction[0]))

        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        sentiment = label_map.get(predicted_label, "Unknown")

        return {
            "original_text": review.text,
            "cleaned_text": clean_text(review.text),
            "prediction": predicted_label,
            "sentiment": sentiment,
            "probabilities_vector": prediction[0].tolist()
        }
    except Exception as e:
        return {"error": str(e)}
