📌 Project Title
Email Spam Detection using Naive Bayes and Streamlit

📝 Description
A machine learning-based web application that classifies emails or messages as Spam or Not Spam using the Naive Bayes algorithm. The project includes a clean user interface built with Streamlit and uses text preprocessing and CountVectorizer to convert raw text into numerical features.

🎯 Objective
The main objective is to develop a user-friendly, real-time classification tool that:

Helps detect spam messages using textual patterns and probabilistic learning.

Applies preprocessing techniques to clean and prepare the data.

Leverages a trained machine learning model for predictions.

🚀 Features
🔍 Classifies input messages as Spam or Not Spam

📊 Preprocessing with CountVectorizer using bag-of-words and stop word removal

🧠 Machine learning model trained with Multinomial Naive Bayes

💻 Interactive and lightweight Streamlit web interface

🧹 Removes duplicates and handles null values in the dataset

🛠️ Technologies Used
Python
Pandas – for data manipulation
Scikit-learn – for ML modeling and text vectorization
Streamlit – for building the web interface
Naive Bayes Classifier – for classification tasks
CountVectorizer – to convert text data into feature vectors

🧠Machine Learning Workflow
Data Collection
The dataset contains labeled messages as ham (not spam) or spam.
Includes message text and corresponding categories.

Data Cleaning and Preprocessing
Removal of duplicate entries.
Handling of missing/null values.
Label renaming (e.g., 'ham' → 'Not Spam') for better user clarity.

Text Vectorization
Uses CountVectorizer to transform raw text into a bag-of-words format.
Stop words (e.g., "and", "the", "is") are removed to reduce noise.

Model Training
Trained using the Multinomial Naive Bayes algorithm, which is effective for classification with discrete features like word counts.
Model is evaluated using an 80-20 train-test split.

Prediction Function
Takes a message as input, processes it using the vectorizer, and predicts the class using the trained model.

📂 How to Run
pip install -r requirements.txt
streamlit run email_spam_detection.py

🔍 Example
Input: "Congratulations! You've won a $1000 gift card. Click here to claim now."
Output: Spam

📈 Accuracy
Model trained on a labeled dataset and evaluated using train_test_split. Accuracy score printed during testing phase that comes out to be 98.74%.

📌 Future Improvements
Add support for email body & subject separately

Use TF-IDF instead of CountVectorizer

Add model performance metrics (precision, recall, confusion matrix)

💡 Key Highlights
Focused on natural language processing (NLP) for email classification.
Implements classic and efficient machine learning techniques.
Demonstrates a complete ML workflow from data preprocessing to deployment.
Easy to use, even for non-technical users via a web interface.

🔐 Real-World Applications
Spam filtering in email clients
SMS spam detection in mobile apps
Chat moderation tools
Fraud detection in messaging systems

🤝 Contribution
Feel free to fork the repository, improve the code, and submit a pull request!
