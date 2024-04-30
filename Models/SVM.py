import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

csv_file = "/content/Data.csv"
data = pd.read_csv(csv_file)

data['sentiment'] = 'neutral'
data.loc[data['positive'] > data['negative'], 'sentiment'] = 'positive'
data.loc[data['negative'] > data['positive'], 'sentiment'] = 'negative'

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=15000)  
X_train = vectorizer.fit_transform(train_df['Text'])
X_test = vectorizer.transform(test_df['Text'])

sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
y_train = train_df['sentiment'].map(sentiment_mapping)
y_test = test_df['sentiment'].map(sentiment_mapping)

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=sentiment_mapping.keys()))
