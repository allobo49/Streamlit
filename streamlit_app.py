import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import nltk
from nltk.corpus import stopwords

# Download the stopwords from NLTK
nltk.download('stopwords')

# Ensure the function that uses NLTK is correctly using the downloaded data
def stopword_proportion(text):
    sw = set(stopwords.words('english'))  # Make sure this is correct; you mentioned French texts
    words = text.split()
    return sum(1 for word in words if word in sw) / len(words)

# Load your trained model
@st.cache(allow_output_mutation=True)  # Use caching to load the model only once
def load_model():
    return load('best_model_LR_features.joblib')  # Adjust the path as needed

model = load_model()

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
    from nltk.corpus import stopwords
    sw = set(stopwords.words('english'))
    words = text.split()
    return sum(1 for word in words if word in sw) / len(words)

def flesch_kincaid_readability(text):
    import textstat
    return textstat.flesch_kincaid_grade(text)

# Create a function to extract features from text
def extract_features(text):
    features = {
        'count_words': count_words(text),
        'avg_word_length': avg_word_length(text),
        'punctuation_count': count_punctuation(text),
        'stopword_proportion': stopword_proportion(text),
        'flesch_kincaid_readability': flesch_kincaid_readability(text),
        # Add the other features extraction as needed
    }
    return pd.DataFrame([features])

# Streamlit app interface
st.title('French4U 🇫🇷')
st.header('Text Difficulty Predictor')
st.write('Enter a French text below and click the button to analyze its difficulty. The difficulty is rated on a scale from 0 to 6, where 0 corresponds to a basic A1 level and 6 denotes proficiency at the C2 level.')

user_input = st.text_area("Insert your text here", height=150)

if st.button("Predict Difficulty"):
    if user_input:  # Correct the variable name here
        # Feature extraction and reshape the input for the model
        features = extract_features(user_input)  # And here
        prediction = model.predict(features)
        st.write(f"Predicted Difficulty Level: {prediction[0]}")
    else:
        st.write("Please enter some text to predict its difficulty.")
