import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Page Configuration
st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: "Arial", sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Data Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Sample Dataset
data = pd.DataFrame({
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

# Preprocess Reviews
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Sidebar Header
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analyze Reviews", "Model Performance"])

# Home Page
if page == "Home":
    st.title("💊 Drug Recommendation System")
    st.markdown("""
        Welcome to the **Drug Recommendation System**. This tool analyzes drug reviews to recommend medications based on **sentiment analysis**.
        The system uses **Machine Learning** techniques and interactive visualizations for accurate recommendations.
        
        ### Features:
        - Analyze reviews and classify them as **Positive** or **Negative**.
        - Get recommendations based on **Positive Sentiments**.
        - View model accuracy and performance metrics.

        Use the **sidebar** to navigate!
    """)

# Analyze Reviews Page
elif page == "Analyze Reviews":
    st.title("📊 Analyze Drug Reviews")
    st.write("Input drug reviews to analyze their sentiment and get recommendations.")

    # Input Section
    user_reviews = st.text_area("Enter Drug Reviews (separate reviews with a newline):", height=150)

    if st.button("Predict Sentiment"):
        if user_reviews.strip():
            # Preprocess and Vectorize Input Reviews
            cleaned_reviews = [preprocess_text(review) for review in user_reviews.split('\n')]
            vectorized_reviews = vectorizer.transform(cleaned_reviews).toarray()

            # Predict Sentiments
            predictions = model.predict(vectorized_reviews)

            # Display Results
            results = pd.DataFrame({
                'Review': user_reviews.split('\n'),
                'Sentiment': ['Positive' if pred == 1 else 'Negative' for pred in predictions]
            })
            st.write("### Sentiment Analysis Results:")
            st.dataframe(results)

            # Show Positive Recommendations
            positive_reviews = results[results['Sentiment'] == 'Positive']
            if not positive_reviews.empty:
                st.write("### Recommended Drugs Based on Positive Reviews:")
                for idx, review in enumerate(positive_reviews['Review'], start=1):
                    st.write(f"{idx}. {review}")
            else:
                st.warning("No positive reviews found.")
        else:
            st.error("Please enter some reviews to analyze.")

# Model Performance Page
elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    st.write("Evaluate the performance of the Machine Learning model used in this system.")

    # Display Model Accuracy
    st.write(f"### Accuracy: **{accuracy*100:.2f}%**")

    # Confusion Matrix
    st.write("### Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # Additional Metrics
    st.markdown("""
        - **Precision**: Measure of how many selected items are relevant.
        - **Recall**: Measure of how many relevant items are selected.
        - **F1-Score**: Harmonic mean of precision and recall.
    """)



