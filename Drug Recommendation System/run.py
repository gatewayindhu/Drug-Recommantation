import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import StringIO

nltk.download('stopwords')

# Data loading and preprocessing functions
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Removing special characters
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Default dataset
def load_default_dataset():
    return pd.DataFrame({
        'review': [
            "This drug works well.",
            "Caused severe side effects.",
            "Very effective for headaches.",
            "Not effective at all.",
            "Had some mild side effects.",
            "Excellent for pain relief!"
        ],
        'sentiment': [1, 0, 1, 0, 1, 1]  # 1 for positive, 0 for negative
    })

# Load the default dataset
def load_custom_dataset(file):
    return pd.read_csv(file)

# Feature extraction and model training
def train_model(data, alpha_value):
    data['cleaned_review'] = data['review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_review']).toarray()
    y = data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB(alpha=alpha_value)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, vectorizer, accuracy

# Word Cloud visualization
def generate_word_cloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Sentiment Prediction with Confidence
def generate_predictions_with_confidence(reviews, vectorizer, model):
    cleaned_reviews = [preprocess_text(review) for review in reviews.split('\n')]
    vectorized_reviews = vectorizer.transform(cleaned_reviews).toarray()  # Vectorizing the input reviews
    probabilities = model.predict_proba(vectorized_reviews)
    
    predictions = []
    for review, prob in zip(reviews.split('\n'), probabilities):
        sentiment = "Positive" if np.argmax(prob) == 1 else "Negative"
        confidence = np.max(prob)  # Confidence is the max probability for the predicted sentiment
        predictions.append({"Review": review, "Sentiment": sentiment, "Confidence": confidence * 100})
    
    return predictions

# Downloadable predictions
def download_predictions(predictions):
    results = pd.DataFrame(predictions)  # Prepare prediction data
    csv = results.to_csv(index=False)
    st.download_button(label="Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# App UI
st.title("Drug Recommendation System based on Sentiment Analysis")
st.write("This system recommends drugs based on sentiment analysis of reviews.")

# Sidebar settings
st.sidebar.header("Upload or Use Default Dataset")
file_upload = st.sidebar.file_uploader("Upload a CSV file with 'review' and 'sentiment' columns", type=['csv'])

# Alpha hyperparameter slider for Naive Bayes model
alpha_value = st.sidebar.slider("Select Alpha for Naive Bayes", 0.01, 2.0, 1.0)

# Handle dataset
if file_upload is not None:
    data = load_custom_dataset(file_upload)
    st.sidebar.success("File uploaded successfully!")
else:
    data = load_default_dataset()
    st.sidebar.warning("Using default dataset.")

# Model training
st.sidebar.text("Training Model...")
with st.spinner('Training the model...'):
    model, vectorizer, accuracy = train_model(data, alpha_value)

# Display the model accuracy
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Show word cloud for positive and negative reviews
if st.checkbox("Show Word Cloud"):
    st.write("Generating Word Cloud for reviews...")
    positive_reviews = data[data['sentiment'] == 1]['review']
    generate_word_cloud(positive_reviews)

# Sentiment Prediction
st.sidebar.header("Input Drug Reviews")
user_reviews = st.sidebar.text_area("Enter drug reviews (separate reviews with a newline):")

# Progress Bar
if st.sidebar.button("Get Predictions"):
    if user_reviews:
        with st.spinner('Generating predictions...'):
            predictions = generate_predictions_with_confidence(user_reviews, vectorizer, model)
            st.write("Predictions for the Reviews:")
            st.write(pd.DataFrame(predictions))  # Show the predictions in a table

            # Provide the option to download predictions
            download_predictions(predictions)
    else:
        st.write("Please enter some reviews to get predictions.")

# Display sentiment distribution chart
sentiment_counts = data['sentiment'].value_counts()
st.bar_chart(sentiment_counts)
