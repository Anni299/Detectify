# naive_bayes_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv("datasetf.csv")

# Split the dataset into features and target variable
X = data['Pattern String']
y = data['Deceptive?']

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
naive_bayes_model = RandomForestClassifier(n_estimators=100, random_state=42)
naive_bayes_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the trained model
joblib.dump(naive_bayes_model, 'naive_bayes_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')