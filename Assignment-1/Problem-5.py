import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Step 1: Create synthetic product feedback data
positive_feedback = [
    "Love this product!", "Excellent quality", "Works perfectly", "Highly recommend it",
    "Very satisfied", "Great value for money", "Exceeded expectations", "Fantastic purchase",
    "Top notch build", "Really good!", "Amazing product", "Very useful", "Easy to use",
    "Reliable and efficient", "Just what I needed", "Perfect for daily use", "Five stars",
    "Would buy again", "Great experience", "Happy with the purchase",
    "Best in the market", "Smooth performance", "User-friendly design", "Great customer support",
    "Everything works as expected", "Affordable and reliable", "High quality material",
    "Very durable", "Awesome features", "Perfect condition",
    "Love the design", "Incredible value", "Best deal", "Fits perfectly", "Looks great",
    "No complaints", "Delivers as promised", "Comfortable to use", "Well packaged", "Timely delivery",
    "Impressive product", "Easy to setup", "Simple and sleek", "Nice finish", "Very impressed",
    "Met all my needs", "Top quality", "Definitely recommend", "Exceptional product", "Thank you!"
]

negative_feedback = [
    "Very disappointed", "Poor quality", "Stopped working", "Would not recommend",
    "Waste of money", "Terrible experience", "Not worth the price", "Cheap material",
    "Doesn't work", "Very noisy", "Looks used", "Misleading description", "Faulty item",
    "Terrible support", "Too expensive", "Never again", "Broken on arrival", "Bad finish",
    "Unacceptable", "Extremely dissatisfied", "Useless product", "Too small", "Not as described",
    "Came late", "No instructions included", "Packaging was damaged", "Broke after one use",
    "Low durability", "Returned immediately", "Difficult to use",
    "Doesn't meet expectations", "Feels cheap", "Very flimsy", "Not reliable", "Defective item",
    "Doesn't fit", "Stopped functioning", "Wrong item delivered", "Uncomfortable to use", "Noisy and slow",
    "Not satisfied", "Feels poorly made", "Doesn't do the job", "Not what I expected",
    "Poor instructions", "Item missing", "Bad packaging", "Doesn’t hold charge", "Awful product", "Wasteful"
]

texts = positive_feedback + negative_feedback
labels = ['good'] * 50 + ['bad'] * 50

df = pd.DataFrame({'Text': texts, 'Label': labels})

# Step 2: Preprocess using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
X = vectorizer.fit_transform(df['Text'])
y = df['Label']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predict and calculate metrics
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, pos_label='good')
recall = recall_score(y_test, y_pred, pos_label='good')
f1 = f1_score(y_test, y_pred, pos_label='good')

print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")

# Step 6: Function to preprocess and vectorize text
def text_preprocess_vectorize(texts, vectorizer):
    """
    Vectorize list of text samples using the fitted TfidfVectorizer.
    
    Parameters:
        texts (list): List of text samples (strings)
        vectorizer (TfidfVectorizer): A fitted TfidfVectorizer object
        
    Returns:
        scipy.sparse matrix: Vectorized representation of the input texts
    """
    return vectorizer.transform(texts)

#  Example usage of the function
sample_texts = ["Great product and value", "Very poor and disappointing"]
sample_vec = text_preprocess_vectorize(sample_texts, vectorizer)
predicted_labels = model.predict(sample_vec)
print("\nSample Predictions:", list(zip(sample_texts, predicted_labels)))
