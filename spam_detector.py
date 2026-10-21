# SMS SPAM DETECTION USING MACHINE LEARNING

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Create a sample dataset
df = pd.read_csv("spam_data_big.csv")
df.columns = ["label", "message"]

# Step 2: Separate messages and labels
X = df["message"]
y = df["label"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Step 4: Convert text into numerical features
vectorizer = CountVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Step 5: Train the Machine Learning model
model = MultinomialNB()
model.fit(X_train_vector, y_train)

# Step 6: Test model accuracy
predictions = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, predictions)

print("===================================")
print("     SMS SPAM DETECTION SYSTEM")
print("===================================")
print(f"Model accuracy on test data: {accuracy * 100:.2f}%")

# Step 7: Predict a new SMS message
print("\nEnter an SMS message to check:")

message = input("SMS: ")

message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)[0]

print("\nResult:")

if prediction == "spam":
    print("WARNING! This message may be SPAM.")
else:
    print("This message is classified as HAM (Normal).")