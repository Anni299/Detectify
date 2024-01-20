# app.py

from flask import Flask, render_template, request
import joblib
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

app = Flask(__name__, static_url_path='/static')

# Load the trained model and vectorizer
naive_bayes_model = joblib.load('naive_bayes_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict_deception(sentence):
    # Convert input text to numerical data using the loaded vectorizer
    sentence_vectorized = vectorizer.transform([sentence])
    # Make a prediction using the loaded model
    prediction = naive_bayes_model.predict(sentence_vectorized)
    return prediction[0], sentence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/FAQIndex.html')
def faqs():
    return render_template('FAQIndex.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        paragraph = request.form['paragraph']
        # Split the paragraph into sentences
        sentences = sent_tokenize(paragraph)
        
        # Make predictions for each sentence
        predictions = [predict_deception(sentence) for sentence in sentences]

        highlighted_paragraph = ""
        for prediction, sentence in predictions:
            if prediction == 1:
                highlighted_paragraph += f'<span style="background-color:  #3fdbdb;">{sentence}</span> '
            else:
                highlighted_paragraph += f'{sentence} '

        return render_template('index.html', paragraph=highlighted_paragraph, predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)
