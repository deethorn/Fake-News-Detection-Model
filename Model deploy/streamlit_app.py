# save this as app.py
import streamlit as st
import pickle, re, gzip

# load artifacts
#model = pickle.load(open("model1.pkl", "rb"))
with gzip.open("model1.pkl.gz", "rb") as f:
    model = pickle.load(f)
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

st.title("Fake News Detection Model")

user_input = st.text_area("Enter a news headline or article text:")
if st.button("Predict"):
    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    st.write("Prediction:", pred)

