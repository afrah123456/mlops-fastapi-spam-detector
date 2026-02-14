"""
Data preparation for spam detection using Kaggle dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def load_spam_data():
    """
    Load spam dataset from CSV file
    """
    # Load the dataset
    df = pd.read_csv('../data/emails.csv')

    print("Available columns:", df.columns.tolist())
    print(f"Total samples: {len(df)}")

    # Your CSV has columns: 'text' and 'spam'
    # Keep only necessary columns
    df = df[['text', 'spam']]
    df.columns = ['text', 'label']

    # Remove any rows with missing values
    df = df.dropna()

    print(f"Spam samples: {df['label'].sum()}")
    print(f"Ham samples: {len(df) - df['label'].sum()}")

    return df

def prepare_data():
    """
    Prepare train-test split
    """
    df = load_spam_data()

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test