import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from joblib import load
import textstat

import nltk
import os

# Set NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download the stopwords corpus
try:
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    print("Error downloading NLTK resources:", e)


# Define the required functions for feature extraction
def count_words(text):
    return len(text.split())

def add_dot(text):
    if not text.endswith('.'):
        text += '.'
    return text

def avg_word_length(text):
    words = text.split()
    return np.mean([len(word) for word in words])

def count_punctuation(text):
    from string import punctuation
    return sum(1 for char in text if char in punctuation)

def stopword_proportion(text):
    sw = set(stopwords.words('french'))
    words = text.split()
    return sum(1 for word in words if word in sw) / len(words)

def flesch_kincaid_readability(text):
    return textstat.flesch_kincaid_grade(text)

def extract_features(text):
    base_features = {
        'sentence':text,
        'count_words': count_words(text),
        'avg_word_length': avg_word_length(text),
        'punctuation_count': count_punctuation(text),
        'stopword_proportion': stopword_proportion(text),
        'flesch_kincaid_readability': flesch_kincaid_readability(text)
    }
    print("Extracted features:", base_features)
    df = pd.DataFrame([base_features])
    return df
    
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
            model = load('best_model_LR_features.joblib')
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
