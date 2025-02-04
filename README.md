# Drug-Recommantation
Drug Recommendation with Sentiment Analysis
3. Hardware & Software Specifications
Hardware Requirements:
Processor: Intel Core i3 or above
RAM: 8 GB or higher
Storage: Minimum 500 GB HDD/SSD
GPU (Optional): NVIDIA GTX 1050 or higher (if using deep learning models)
Software Requirements:
Operating System: Windows/Linux/MacOS
Programming Language: Python (v3.7 or later)
Libraries/Frameworks:
Streamlit (for UI)
Pandas, NumPy, Scikit-learn (for data processing and ML)
NLTK (for text preprocessing)
TfidfVectorizer (feature extraction)
MultinomialNB (Naive Bayes classifier)
Development Environment: Jupyter Notebook, VS Code, or PyCharm
4. Modules with Descriptions
1. Data Collection:
Description: Collect drug reviews and sentiment labels from open-source datasets or user-uploaded files.
Purpose: Provides the raw data for analysis.
2. Data Preprocessing:
Description: Clean data by removing special characters, converting text to lowercase, and filtering stopwords using NLTK.
Purpose: Prepare the text data for feature extraction.
3. Feature Extraction:
Description: Convert preprocessed text into numerical form using TF-IDF vectorization.
Purpose: Enable machine learning models to interpret the text data.
4. Model Training and Evaluation:
Description: Train the Naive Bayes classifier on the TF-IDF features and evaluate performance using metrics like accuracy.
Purpose: Develop a reliable model for sentiment analysis.
5. Sentiment Prediction and Recommendation:
Description: Accept user input reviews, predict sentiments, and recommend drugs based on positive reviews.
Purpose: Provide personalized drug recommendations to users.
6. User Interface:
Description: Use Streamlit to build an interactive dashboard for uploading datasets, visualizing results, and getting recommendations.
Purpose: Enhance user experience with a clean and responsive UI.
5. Existing & Proposed System
Existing System:
Focuses on manual review of drug effectiveness and side effects from online platforms.
Limitations:
Time-consuming and inefficient.
No automated sentiment analysis for drug recommendations.
Lack of real-time feedback for users.
Proposed System:
Automates drug review analysis using machine learning and natural language processing.
Features:
Sentiment analysis of drug reviews to predict positive and negative sentiments.
Recommendations based on user-provided reviews.
Interactive dashboard for uploading datasets and displaying results.
Advantages:
Faster and more accurate analysis.
Real-time drug recommendations based on sentiment insights.
Enhanced decision-making for drug efficacy.

**Run**
streamlit run app.py
pip install......requirements
