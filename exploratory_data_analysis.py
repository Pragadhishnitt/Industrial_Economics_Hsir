import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

csv_file = "/content/data(2).csv"
data = pd.read_csv(csv_file)

# 1. Sentiment Score Distribution Visualization (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data['positive'], bins=20, color='green', kde=True, label='Positive')
sns.histplot(data['negative'], bins=20, color='red', kde=True, label='Negative')
sns.histplot(data['neutral'], bins=20, color='blue', kde=True, label='Neutral')
sns.histplot(data['compound'], bins=20, color='orange', kde=True, label='Compound')
plt.title('Sentiment Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 2. Text Length Distribution Visualization (Histogram)
data['text_length'] = data['Text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data, x='text_length', kde=True, bins=20)
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# 3. Bi-gram Frequency Visualization (Bar Chart)
ngram_counts = Counter([tuple(text.split()[i:i+2]) for text in data['Text'] for i in range(len(text.split())-1)])
most_common_ngrams = ngram_counts.most_common(10)
ngrams = [ngram for ngram, _ in most_common_ngrams]
counts = [count for _, count in most_common_ngrams]

plt.figure(figsize=(10, 6))
plt.bar(range(len(ngrams)), counts)
plt.xticks(range(len(ngrams)), ngrams, rotation=45)
plt.title('Top 10 Bi-gram Frequencies')
plt.xlabel('Bi-gram')
plt.ylabel('Frequency')
plt.show()

# 4. Distribution of Positive, Negative, Neutral, and Compound Scores (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['positive', 'negative', 'neutral', 'compound']])
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Scores')
plt.ylabel('Score Value')
plt.show()

# 5. Pairwise Scatterplot for Sentiment Scores
plt.figure(figsize=(10, 6))
sns.pairplot(data[['positive', 'negative', 'neutral', 'compound']])
plt.title('Pairwise Scatterplot for Sentiment Scores')
plt.show()

# 6. Heatmap of Sentiment Scores
plt.figure(figsize=(8, 6))
sns.heatmap(data[['positive', 'negative', 'neutral', 'compound']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Sentiment Scores')
plt.show()

# 7. Violin Plot for Sentiment Scores
plt.figure(figsize=(10, 6))
sns.violinplot(data=data[['positive', 'negative', 'neutral', 'compound']], inner="quartile")
plt.title('Violin Plot of Sentiment Scores')
plt.xlabel('Sentiment Scores')
plt.ylabel('Score Value')
plt.show()

# 8. Stacked Bar Chart of Sentiment Scores
sentiment_counts = data[['positive', 'negative', 'neutral']].sum()
plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='bar', stacked=True, color=['green', 'red', 'blue'])
plt.title('Stacked Bar Chart of Sentiment Scores')
plt.xlabel('Sentiment')
plt.ylabel('Total Score')
plt.show()
