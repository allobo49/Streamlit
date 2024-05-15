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

# Function to map difficulty levels to CEFR proficiency levels
def map_to_cefr(difficulty_level):
    cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    return cefr_levels[difficulty_level]
    
# Streamlit app interface
st.title('French4U 🇫🇷')
st.header('Text Difficulty Predictor')
st.write('Enter a French sentence below and click the button to analyze its difficulty.')
st.write('Note that if the sentence is not in French, the difficulty prediction will not be accurate.')
user_input = st.text_area("Insert your text here", height=150)

if st.button("Predict Difficulty"):
    if user_input:
        try:
            model = load('best_model_LR_features.joblib')
            if model:
                features = extract_features(user_input)
                prediction = model.predict(features)
                predicted_cefr = map_to_cefr(prediction[0])
                st.write(f"Predicted difficulty level: {predicted_cefr}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.write("Please enter some text to predict its difficulty.")
