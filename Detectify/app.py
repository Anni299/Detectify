# app.py

# Import necessary libraries and modules
from flask import Flask, render_template, request
import joblib
import nltk
from nltk.tokenize import sent_tokenize

# Download the punkt module from NLTK for sentence tokenization
nltk.download('punkt')

# Initialize the Flask app and define the static URL path
app = Flask(__name__, static_url_path='/static')

# Load the trained Naive Bayes model and vectorizer from joblib files
naive_bayes_model = joblib.load('naive_bayes_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Function to predict deception for a given sentence
def predict_deception(sentence):
    # Convert the input text to numerical data using the loaded vectorizer
    sentence_vectorized = vectorizer.transform([sentence])
    
    # Make a prediction using the loaded Naive Bayes model
    predi
