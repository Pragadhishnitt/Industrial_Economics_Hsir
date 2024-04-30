import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
train_labels = torch.tensor(train_df['sentiment'].map(sentiment_mapping).values)
test_labels = torch.tensor(test_df['sentiment'].map(sentiment_mapping).values)
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=sentiment_mapping.keys()))
