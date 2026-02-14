"""
Train spam detection model
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from data import prepare_data


def train_model():
    """
    Train spam detection model using Naive Bayes
    """
    print("Loading data...")
    X_train, X_test, y_train, y_test = prepare_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Create pipeline with TF-IDF vectorizer and Naive Bayes classifier
    print("\nTraining model...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('classifier', MultinomialNB())
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Ham', 'Spam']))

    # Save the model
    print("\nSaving model...")
    with open('../model/spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model saved successfully as 'spam_model.pkl'")

    return model


if __name__ == "__main__":
    train_model()