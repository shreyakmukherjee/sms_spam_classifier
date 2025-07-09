import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Preprocessing function (no punkt)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)  # regex word tokenization
    text = [word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load the trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message here")

if st.button("Predict"):
    # 1. Transform
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display Result
    if result == 1:
        st.error("🚫 This message is classified as **SPAM**.")
    else:
        st.success("✅ This message is classified as **NOT SPAM**.")
