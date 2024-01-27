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
    prediction = naive_bayes_model.predict(sentence_vectorized)
    
    # Return the prediction result and the original sentence
    return prediction[0], sentence

# Define the route for the home page
@app.route('/')
def index():
    # Render the HTML template for the home page
    return render_template('index.html')

# Define the route for the FAQ page
@app.route('/FAQIndex.html')
def faqs():
    # Render the HTML template for the FAQ page
    return render_template('FAQIndex.html')

# Define the route for making predictions based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request method is POST
    if request.method == 'POST':
        # Get the user input paragraph from the form
        paragraph = request.form['paragraph']
        
        # Split the paragraph into individual sentences
        sentences = sent_tokenize(paragraph)
        
        # Make predictions for each sentence using the predict_deception function
        predictions = [predict_deception(sentence) for sentence in sentences]

        # Initialize an empty string to store the highlighted paragraph
        highlighted_paragraph = ""

        # Loop through the predictions and sentences to generate the highlighted paragraph
        for prediction, sentence in predictions:
            if prediction == 1:
                # If the prediction indicates deception, add a highlighted span to the sentence
                highlighted_paragraph += f'<span style="background-color:  #3fdbdb;">{sentence}</span> '
            else:
                # If no deception is predicted, add the original sentence
                highlighted_paragraph += f'{sentence} '

        # Render the home page template with the highlighted paragraph and predictions
        return render_template('index.html', paragraph=highlighted_paragraph, predictions=predictions)

# Run the Flask app if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)

