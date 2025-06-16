import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Create a synthetic dataset
positive_reviews = [
    "Loved it!", "Amazing movie", "Great plot and acting", "Fantastic direction", "Highly recommend",
    "Superb cinematography", "Very entertaining", "Outstanding!", "A masterpiece", "Just brilliant",
    "Loved the cast", "Powerful story", "Excellent film", "Top-notch acting", "Really good!",
    "Incredible movie", "Heartwarming", "Fun and exciting", "Fabulous!", "Worth watching",
    "Marvelous experience", "Delightful!", "Inspirational", "Breathtaking scenes", "Beautifully made",
    "Very touching", "Great soundtrack", "So moving", "Absolutely loved it", "Perfect!",
    "Oscar-worthy", "Highly enjoyable", "Classic!", "Emotionally rich", "Best movie this year",
    "Well directed", "Captivating", "Smart and funny", "Loved the pacing", "Truly remarkable",
    "Top class", "Will watch again", "Uplifting", "Fun ride", "Super engaging", "Loved the script",
    "Flawless", "Can't stop thinking about it", "Phenomenal", "Feel good movie"
]

negative_reviews = [
    "Terrible movie", "Boring and slow", "Awful plot", "Waste of time", "Poor acting",
    "Hated it", "Too long", "Really bad", "What a mess", "Disappointing",
    "Unwatchable", "Not worth it", "Terribly written", "Very dull", "Lame!",
    "Cliché and boring", "Uninspired", "Poorly made", "Forgettable", "Zero emotion",
    "Ridiculous story", "Bad editing", "Painful to watch", "Super predictable", "I regret watching it",
    "Not engaging", "Fails completely", "Slow and weak", "Too confusing", "Cringe!",
    "Weak performances", "Unbelievable characters", "Lazy writing", "Terrible pacing", "Flat story",
    "Horrible", "Lacked substance", "Disjointed", "I slept off", "Terrible ending",
    "A total flop", "Poor direction", "Worst film ever", "Dreadful", "Shallow and empty",
    "Really annoying", "Disaster", "Broken script", "Didn't enjoy it", "Not good at all"
]

reviews = positive_reviews + negative_reviews
sentiments = ['positive'] * 50 + ['negative'] * 50

df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

# Step 2: Vectorize reviews using CountVectorizer
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")

# Step 6: Predict function
def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    return model.predict(review_vector)[0]

# ✅ Example use of predict_review_sentiment
example = "The movie was fantastic with amazing visuals!"
print("Predicted Sentiment:", predict_review_sentiment(model, vectorizer, example))
