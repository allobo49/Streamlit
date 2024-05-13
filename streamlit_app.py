import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from joblib import load

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = load('best_model_LR_features.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# NLTK resources for stopwords
nltk.download('stopwords')

# Define the required functions for feature extraction
def extract_features(text):
    words = text.split()
    num_words = len(words)
    avg_word_length = np.mean([len(word) for word in words])
    punctuation_count = sum(1 for char in text if char in set(',.:;!?'))
    sw = set(stopwords.words('french'))
    stopword_proportion = sum(1 for word in words if word in sw) / num_words
    return pd.DataFrame([[num_words, avg_word_length, punctuation_count, stopword_proportion]],
                        columns=['count_words', 'avg_word_length', 'punctuation_count', 'stopword_proportion'])

# Streamlit app interface
st.title('French4U 🇫🇷')
st.header('Text Difficulty Predictor')
st.write('Enter a French text below and click the button to analyze its difficulty.')

user_input = st.text_area("Insert your text here", height=150)

if st.button("Predict Difficulty"):
    st.write("Button clicked!")
    if user_input:
        st.write("Text provided:", user_input)
        try:
            model = load_model()
            if model:
                st.write("Model loaded!")
                features = extract_features(user_input)
                st.write("Features extracted:", features)
                prediction = model.predict(features)
                st.write(f"Predicted Difficulty Level: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.write("Please enter some text to predict its difficulty.")
