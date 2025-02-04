import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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
data = load_default_dataset()

# Apply text preprocessing
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar: Model selection
st.title("Drug Recommendation System based on Sentiment Analysis")
st.write("This system recommends drugs based on sentiment analysis of reviews.")

st.sidebar.header("Select Classification Model")
model_choice = st.sidebar.selectbox("Choose a model", ["Naive Bayes", "Logistic Regression", "SVM"])

# Model selection based on user input
if model_choice == "Naive Bayes":
    model = MultinomialNB()
elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = SVC()

# Model training
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display the model accuracy
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Confusion matrix
def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

plot_confusion_matrix()

# Word Cloud
def plot_wordcloud():
    text = ' '.join(data['cleaned_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.header("Word Cloud for Reviews")
    st.image(wordcloud.to_array())

plot_wordcloud()

# Sentiment Prediction Confidence
def display_prediction_confidence(reviews):
    st.write("Sentiment Prediction with Confidence:")
    cleaned_reviews = [preprocess_text(review) for review in reviews.split('\n')]
    vectorized_reviews = vectorizer.transform(cleaned_reviews).toarray()  # Ensure this is defined here
    probabilities = model.predict_proba(vectorized_reviews)
    
    predictions = []
    for review, prob in zip(reviews.split('\n'), probabilities):
        sentiment = "Positive" if np.argmax(prob) == 1 else "Negative"
        confidence = np.max(prob)
        predictions.append({"Review": review, "Sentiment": sentiment, "Confidence": confidence * 100})
    
    return predictions

# Text input for reviews
st.sidebar.header("Input Drug Reviews")
user_reviews = st.sidebar.text_area("Enter drug reviews (separate reviews with a newline):")

if st.sidebar.button("Get Recommendations"):
    if user_reviews:
        # Preprocess and vectorize the input reviews
        cleaned_reviews = [preprocess_text(review) for review in user_reviews.split('\n')]
        vectorized_reviews = vectorizer.transform(cleaned_reviews).toarray()
        
        # Predict sentiment for each review
        predictions = model.predict(vectorized_reviews)
        
        # Filter positive reviews
        recommended_drugs = [review for review, sentiment in zip(user_reviews.split('\n'), predictions) if sentiment == 1]
        
        if recommended_drugs:
            st.write("Recommended Drugs based on positive reviews:")
            for drug in recommended_drugs:
                st.write(f"- {drug}")
        else:
            st.write("No positive reviews found. Consider trying different drugs.")
    else:
        st.write("Please enter some reviews to get recommendations.")

# Display sentiment distribution
def plot_sentiment_distribution():
    sentiment_counts = data['sentiment'].value_counts()
    sentiment_labels = ['Positive', 'Negative']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sentiment_labels, sentiment_counts, color=['green', 'red'])
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    
    st.pyplot(fig)

plot_sentiment_distribution()

# Display the dataset
def display_dataset():
    st.header("Dataset Preview")
    st.write(data.head())

if st.checkbox("Show Dataset"):
    display_dataset()

# Downloadable predictions
def download_predictions(predictions, reviews):
    results = pd.DataFrame(predictions)  # Prepare prediction data
    csv = results.to_csv(index=False)
    st.download_button(label="Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

if st.sidebar.button("Download Predictions"):
    if user_reviews:
        # Get predictions with confidence
        predictions = display_prediction_confidence(user_reviews)
        download_predictions(predictions, user_reviews)
