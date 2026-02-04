from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

app = Flask(__name__)
CORS(app)  

# --- Load the Model Assets ---
# We use joblib to load the 'memory' of the trained model
MODEL_PATH = 'model_assets/sentiment_model.pkl'
VECTORIZER_PATH = 'model_assets/tfidf_vectorizer.pkl'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --- Preprocessing Function ---
# This must be identical to the one in Jupyter Notebook
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    tokens = word_tokenize(text) # Split into words
    tokens = [word for word in tokens if word not in stop_words] # Remove 'the', 'is', etc.
    return ' '.join(tokens)

# --- 3. The Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text sent from the UI
        data = request.json
        user_input = data.get('text', '')

        if not user_input:
            return jsonify({'error': 'No text provided'}), 400

        # Clean, Vectorize, and Predict
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Return the result as JSON to the UI
        return jsonify({
            'sentiment': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on http://127.0.0.1:5000
    app.run(debug=True)