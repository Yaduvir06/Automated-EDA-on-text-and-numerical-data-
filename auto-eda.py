# Step 1: Install Required Libraries
# Run these commands in your terminal:

# Step 2: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import shap
import xgboost as xgb
import gc

# Define any specific processing steps for each chunk in a function:
def process_chunk(chunk):
    # --- Optimize numeric columns ---
    for col in chunk.select_dtypes(include=["float"]).columns:
        chunk[col] = chunk[col].astype("float32")
    for col in chunk.select_dtypes(include=["int"]).columns:
        chunk[col] = chunk[col].astype("int32")
        
    # --- Convert object columns to 'category' ---
    for col in chunk.select_dtypes(include=["object"]).columns:
        chunk[col] = chunk[col].astype("category")
        
    # --- Process date columns (example: ReadingDateTime) ---
    date_columns = ['ReadingDateTime']  # adjust names as needed
    for col in date_columns:
        if col in chunk.columns:
            chunk[col] = pd.to_datetime(chunk[col], errors='coerce',format='%d/%m/%Y %H:%M')
            # Optionally extract date features:
            chunk[col + "_year"] = chunk[col].dt.year
            chunk[col + "_month"] = chunk[col].dt.month
            chunk[col + "_day"] = chunk[col].dt.day
            chunk[col + "_weekday"] = chunk[col].dt.weekday
            
    # --- Fill missing numeric values (if desired) ---
    chunk.fillna(chunk.mean(numeric_only=True), inplace=True)
    
    # --- Drop duplicates ---
    chunk = chunk.drop_duplicates()
    
    return chunk

# Read and process the data in batches
chunks = []
chunksize = 10000  # Adjust based on your memory capacity

for chunk in pd.read_csv("LaqnData.csv", chunksize=chunksize):
    processed_chunk = process_chunk(chunk)
    chunks.append(processed_chunk)
    gc.collect()  # free memory after processing each chunk

# Concatenate all processed chunks into one DataFrame
df = pd.concat(chunks, ignore_index=True)
print("Full dataset processed. Number of rows:", len(df))
# Convert all object-type columns to string (for consistency in text processing)
categorical_cols = df.select_dtypes(include=["object"]).columns
df[categorical_cols] = df[categorical_cols].astype(str)

# Step 5: Convert Text Data to Numerical Format
# -- 1. Label Encoding for categorical text columns.
le = LabelEncoder()
for col in categorical_cols:
    df[col + "_encoded"] = le.fit_transform(df[col])
    
# -- 2. One-Hot Encoding for categorical text columns.
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -- 3. TF-IDF for a specific text column (replace 'your_text_column' with your actual column name).
text_column = "Species"  # Change this to the name of your text column.
if text_column in df.columns:
    tfidf = TfidfVectorizer(max_features=50)  # Limit to top 100 words.
    tfidf_matrix = tfidf.fit_transform(df[text_column].astype(str))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    df = pd.concat([df, tfidf_df], axis=1)

# -- 4. Word Embeddings using Word2Vec for the same text column.
if text_column in df.columns:
    # Tokenize the text into words.
    df["tokenized_text"] = df[text_column].astype(str).apply(lambda x: x.split())
    # Train a Word2Vec model on the tokenized text.
    word2vec_model = Word2Vec(sentences=df["tokenized_text"], vector_size=100, window=5, min_count=1, workers=4)
    # Compute the average Word2Vec vector for each document.
    df["word2vec_vector"] = df["tokenized_text"].apply(
        lambda tokens: np.mean([word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
                               or [np.zeros(100)], axis=0)
    )

# Step 6: Generate Automated EDA Reports Using Sweetviz
# Sweetviz creates an HTML report summarizing the dataset.
report = sv.analyze(df)
report.show_html("sweetviz_report.html")
print("Sweetviz report generated as 'sweetviz_report.html'.")

# Step 7: Visualization & Insights

# 7.1: Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 7.2: Word Cloud for the text column (if available)
if text_column in df.columns:
    text_data = " ".join(df[text_column].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud for " + text_column)
    plt.show()

# 7.3: Sentiment Analysis on the text column (if available)
if text_column in df.columns:
    df["sentiment"] = df[text_column].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment_category"] = df["sentiment"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
    print("Sentiment category counts:")
    print(df["sentiment_category"].value_counts())

# Step 8: Feature Importance Using SHAP and XGBoost
# Replace 'target_column' with the name of your target variable.
from xgboost import XGBClassifier
target_column = "Provisional or Ratified_R"
if target_column in df.columns:
    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = X.drop(columns=["Site", "ReadingDateTime"], errors="ignore")
    
    # Handle missing values
    X = X.fillna(0)
    
    # Train XGBoost
    model = XGBClassifier()
    model.fit(X, y)
    
    # Use XGBoost's built-in feature importance plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xticks(range(len(X.columns)), X.columns, rotation=90)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()
else:
    print("Target column not found. Skipping feature importance step.")