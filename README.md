Sentiment Analysis Web Application 🤖

This repository contains a full-stack implementation of a Natural Language Processing (NLP) system. The project demonstrates the complete pipeline from data enhancement and model training to deployment via a web interface.
🚀 Overview

The goal of this project was to create a functional tool that can classify user text into three categories: Positive, Negative, or Neutral. Originally developed as a laboratory experiment for Intelligent Systems, it showcases how machine learning models can be transitioned from static notebooks to interactive web services.
Key Features

    Enhanced NLP Dataset: A custom dataset expanded with 50+ additional rows to cover complex edge cases and varied sentiments.
    Machine Learning Pipeline: Utilizes TF-IDF Vectorization and a Multinomial Naive Bayes classifier for high-efficiency text analysis.
    RESTful API: A Flask backend that handles preprocessing and model inference.
    Interactive UI: A clean HTML/CSS/JavaScript frontend that communicates with the backend in real-time.
🛠️ Technical Stack

    Language: Python 3.12

    ML Libraries: Scikit-Learn, Pandas, NLTK

    Web Framework: Flask, Flask-CORS

    Frontend: HTML5, CSS3, JavaScript (Fetch API)
    
📂 Project Structure
Sentiment_Analysis_Project/
├── model_assets/                 # Serialized model artifacts
│   ├── sentiment_model.pkl       # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl      # Saved TF-IDF Vectorizer
├── venv/                         # Python Virtual Environment
├── app.py                        # Flask Web API (Backend)
├── index.html                    # User Interface (Frontend)
├── sentiment_dataset.csv         # Expanded dataset for training
├── Sentiment_Analysis.ipynb      # Training and Export Notebook
└── README.md                     # Documentation

⚙️ How to Run

  Initialize Environment:
    python -m venv venv
    .\venv\Scripts\activate
    
  Install Dependencies:
    pip install flask flask-cors joblib nltk scikit-learn pandas

  Start the Backend:
    python app.py

  Access the UI: Open index.html in any modern web browser while the backend is running.

👨‍💻 Author
Drachir Carlo Tacal
