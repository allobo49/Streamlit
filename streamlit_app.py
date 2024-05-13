import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import textstat
from joblib import load
import spacy
import os

# Specify the path to the model directory
model_path = '/path/to/fr_core_news_sm'

# Check if the model directory exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory '{model_path}' not found.")

# Check if the model directory is readable
if not os.access(model_path, os.R_OK):
    raise PermissionError(f"Insufficient permissions to read model directory '{model_path}'.")

# Load the French language model from SpaCy
try:
    nlp = spacy.load(model_path)
except IOError as e:
    raise IOError(f"An IO error occurred while loading the model: {e}")
except Exception as e:
    raise Exception(f"An error occurred while loading the model: {e}")

# Load your trained model
@st.cache(allow_output_mutation=True)  # Use caching to load the model only once
def load_model():
    return load('best_model_LR_features.joblib')

model = load_model()

# NLTK resources for stopwords
nltk.download('stopwords')
nltk.download('punkt')

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

def analyze_pos(text):
    doc = nlp(text)
    pos_counts = {}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
        else:
            pos_counts[token.pos_] = 1
    return pos_counts

def extract_features(text):
    base_features = {
        'count_words': count_words(text),
        'avg_word_length': avg_word_length(text),
        'punctuation_count': count_punctuation(text),
        'stopword_proportion': stopword_proportion(text),
        'flesch_kincaid_readability': flesch_kincaid_readability(text)
    }
    pos_features = analyze_pos(text)
    features = {**base_features, **pos_features}
    df = pd.DataFrame([features])
    return df

# Streamlit app interface
st.title('French4U 🇫🇷')
st.header('Text Difficulty Predictor')
st.write('Enter a French text below and click the button to analyze its difficulty.')

user_input = st.text_area("Insert your text here", height=150)

if st.button("Predict Difficulty"):
    if user_input:
        try:
            features = extract_features(user_input)
            prediction = model.predict(features)
            st.write(f"Predicted Difficulty Level: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.write("Please enter some text to predict its difficulty.")
