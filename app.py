# ============================================================
# FAKE NEWS DETECTOR - WEB APPLICATION
# How to run: python app.py
# Then open your browser and go to: http://127.0.0.1:5000
# ============================================================

import pickle
import re
import os
from flask import Flask, render_template, request, jsonify
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)

# -----------------------------------------------
# Load the trained model
# -----------------------------------------------
if not os.path.exists('model.pkl'):
    print("ERROR: model.pkl not found!")
    print("Please run 'python train_model.py' first.")
    exit()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# -----------------------------------------------
# Same text cleaning function as in training
# -----------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# -----------------------------------------------
# Main page - shows the website
# -----------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------------------------
# Prediction API - called when user submits text
# -----------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('text', '').strip()

    if not news_text:
        return jsonify({'error': 'Please enter some text.'})

    if len(news_text.split()) < 5:
        return jsonify({'error': 'Please enter at least a sentence (5+ words).'})

    # Clean and predict
    cleaned = clean_text(news_text)
    prediction = model.predict([cleaned])[0]
    
    # Get confidence score (probability)
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.classes_
    
    # Build confidence dict
    confidence_dict = dict(zip(classes, probabilities))
    
    if prediction == 'REAL':
        confidence = confidence_dict.get('REAL', 0.5)
    else:
        confidence = confidence_dict.get('FAKE', 0.5)

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence * 100, 1),
        'fake_prob': round(confidence_dict.get('FAKE', 0) * 100, 1),
        'real_prob': round(confidence_dict.get('REAL', 0) * 100, 1),
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Fake News Detector is running!")
    print("Open your browser and go to:")
    print("  http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True)
