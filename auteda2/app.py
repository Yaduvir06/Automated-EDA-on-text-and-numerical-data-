from flask import Flask, render_template, request, redirect, url_for, flash, send_file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import os
import gc
import psutil
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "data_analysis_app_secret_key"

os.makedirs('static/outputs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)

@app.template_filter('thousands_separator')
def thousands_separator(value):
    return "{:,}".format(value) if value is not None else "0"

@app.route('/')
def home():
    return render_template('index.html')

def log_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    print(f"Memory Usage ({message}): {memory_mb:.2f} MB")
    return memory_mb

def clear_memory():
    plt.close('all')
    gc.collect()

def process_chunk(chunk, date_columns, date_format, text_columns):
    # --- Date columns: split into new features
    for col in date_columns:
        if col in chunk.columns:
            try:
                chunk[col] = pd.to_datetime(chunk[col], errors='coerce', format=date_format)
                chunk[f"{col}_year"] = chunk[col].dt.year
                chunk[f"{col}_month"] = chunk[col].dt.month
                chunk[f"{col}_day"] = chunk[col].dt.day
                chunk[f"{col}_weekday"] = chunk[col].dt.weekday
            except Exception as e:
                print(f"Error processing date column {col}: {e}")

    # --- Encode text columns in-place (replace with label encoding)
    le = LabelEncoder()
    for col in text_columns:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna('MISSING').astype(str)
            chunk[col] = le.fit_transform(chunk[col])

    # --- Handle missing values after encoding
    num_cols = chunk.select_dtypes(include=np.number).columns
    chunk[num_cols] = chunk[num_cols].fillna(chunk[num_cols].median())
    cat_cols = chunk.select_dtypes(include=['category', object]).columns
    for col in cat_cols:
        if not chunk[col].mode().empty:
            chunk[col] = chunk[col].fillna(chunk[col].mode()[0])
        else:
            chunk[col] = chunk[col].fillna('MISSING')

    # --- Drop duplicates after encoding and filling
    chunk = chunk.drop_duplicates()
    return chunk

class DataAnalyzer:
    def __init__(self):
        self.dataset_info = {
            'rows': 0,
            'columns': 0,
            'memory_usage': 0,
            'numeric_columns': 0,
            'categorical_columns': 0
        }
        self.sentiment_results = []
        self.wordcloud_paths = []
        self.heatmap_path = None
        self.feature_importance_path = None
        self.sweetviz_report_path = None
        self.processed_file_path = None

    def update_dataset_info(self, df):
        self.dataset_info['rows'] += len(df)
        self.dataset_info['columns'] = len(df.columns)
        self.dataset_info['memory_usage'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.dataset_info['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)

    def create_sweetviz_report(self, df, sample_size=10000):
        report_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        report = sv.analyze(report_df)
        self.sweetviz_report_path = os.path.join('static', 'outputs', 'sweetviz_report.html')
        report.show_html(self.sweetviz_report_path)
        self.sweetviz_report_path = os.path.basename(self.sweetviz_report_path)
        clear_memory()

    def create_correlation_heatmap(self, df, target_column=None):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 30:
            if target_column in numeric_df.columns:
                corr_with_target = numeric_df.corr()[target_column].abs().sort_values(ascending=False)
                top_cols = corr_with_target.index[:30]
                numeric_df = numeric_df[top_cols]
            else:
                numeric_df = numeric_df.iloc[:, :30]
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        self.heatmap_path = os.path.join('static', 'outputs', 'heatmap.png')
        plt.savefig(self.heatmap_path)
        plt.close()
        self.heatmap_path = os.path.basename(self.heatmap_path)
        clear_memory()

    def process_text_data(self, df, text_columns):
        for text_column in text_columns:
            if text_column in df.columns:
                try:
                    text_data = df[text_column].sample(5000, random_state=42) if len(df) > 5000 else df[text_column]
                    text_data = text_data.fillna('').astype(str)
                    text_string = " ".join(text_data)
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_string)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    plt.title(f"Word Cloud for {text_column}")
                    wc_path = os.path.join('static', 'outputs', f'wordcloud_{text_column}.png')
                    plt.savefig(wc_path)
                    plt.close()
                    self.wordcloud_paths.append((text_column, os.path.basename(wc_path)))
                    sentiments = [TextBlob(text).sentiment.polarity for text in text_data]
                    sentiment_categories = ["Positive" if s > 0 else "Negative" if s < 0 else "Neutral" for s in sentiments]
                    sentiment_counts = pd.Series(sentiment_categories).value_counts().to_dict()
                    self.sentiment_results.append((text_column, sentiment_counts))
                except Exception as e:
                    print(f"Error processing text column {text_column}: {e}")
                    flash(f"Warning: Could not process text data for column {text_column}. Error: {str(e)}")
                clear_memory()

    def analyze_feature_importance(self, df, target_column):
        if target_column not in df.columns:
            flash(f"Target column '{target_column}' not found for feature importance analysis.")
            return
        try:
            X = df.drop(columns=[target_column], errors='ignore')
            y = df[target_column]
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)
            X = X.fillna(0)
            X = X.select_dtypes(include=[np.number])
            if len(X) > 10000:
                X_sample = X.sample(10000, random_state=42)
                y_sample = y[X_sample.index]
            else:
                X_sample = X
                y_sample = y
            unique_count = len(np.unique(y_sample))
            if unique_count <= 10:
                model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
                model_type = "Classifier"
            else:
                model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
                model_type = "Regressor"
            model.fit(X_sample, y_sample)
            plt.figure(figsize=(12, 8))
            if len(X_sample.columns) > 30:
                importances = pd.Series(model.feature_importances_, index=X_sample.columns)
                importances = importances.sort_values(ascending=False)[:30]
                plt.bar(range(len(importances)), importances.values)
                plt.xticks(range(len(importances)), importances.index, rotation=90)
            else:
                plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
                plt.xticks(range(len(X_sample.columns)), X_sample.columns, rotation=90)
            plt.title(f'XGBoost {model_type} Feature Importance')
            plt.tight_layout()
            self.feature_importance_path = os.path.join('static', 'outputs', 'feature_importance.png')
            plt.savefig(self.feature_importance_path)
            plt.close()
            self.feature_importance_path = os.path.basename(self.feature_importance_path)
        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            flash(f"Warning: Could not perform feature importance analysis. Error: {str(e)}")
        clear_memory()

    def save_processed_dataset(self, df, filename):
        processed_path = os.path.join('processed', f'processed_{filename}')
        df.to_csv(processed_path, index=False)
        self.processed_file_path = processed_path
        return processed_path

@app.route('/process', methods=['POST'])
def process_data():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    date_columns = request.form.get('date_columns', '').split(',')
    date_columns = [col.strip() for col in date_columns if col.strip()]
    date_format = request.form.get('date_format', '%d/%m/%Y %H:%M')
    text_columns = request.form.get('text_columns', '').split(',')
    text_columns = [col.strip() for col in text_columns if col.strip()]
    target_column = request.form.get('target_column', '').strip()
    drop_columns = request.form.get('drop_columns', '').split(',')
    drop_columns = [col.strip() for col in drop_columns if col.strip()]
    chunksize = int(request.form.get('chunksize', 10000))

    analyzer = DataAnalyzer()

    try:
        log_memory_usage("Before processing")
        sample_df = pd.read_csv(filepath, nrows=5)
        print("Original column names:", sample_df.columns.tolist())
        first_chunk = next(pd.read_csv(filepath, chunksize=chunksize))
        processed_first_chunk = process_chunk(first_chunk, date_columns, date_format, text_columns)
        if drop_columns:
            processed_first_chunk = processed_first_chunk.drop(columns=drop_columns, errors='ignore')
        full_df = processed_first_chunk.copy()
        analyzer.update_dataset_info(full_df)
        chunk_count = 1
        for chunk in pd.read_csv(filepath, chunksize=chunksize, skiprows=range(1, chunksize+1)):
            chunk_count += 1
            if chunk_count % 5 == 0:
                print(f"Processing chunk {chunk_count}...")
                log_memory_usage(f"During processing chunk {chunk_count}")
            processed_chunk = process_chunk(chunk, date_columns, date_format, text_columns)
            if drop_columns:
                processed_chunk = processed_chunk.drop(columns=drop_columns, errors='ignore')
            full_df = pd.concat([full_df, processed_chunk], ignore_index=True)
            analyzer.update_dataset_info(processed_chunk)
            del processed_chunk
            del chunk
            clear_memory()
            if full_df.memory_usage(deep=True).sum() / (1024 * 1024) > 1000:
                analyzer.process_text_data(full_df, text_columns)
                if target_column in full_df.columns:
                    analyzer.analyze_feature_importance(full_df, target_column)
                if len(full_df.columns) > 50:
                    if target_column in full_df.columns:
                        numeric_cols = full_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            corr_with_target = full_df[numeric_cols].corr()[target_column].abs().sort_values(ascending=False)
                            top_cols = corr_with_target.index[:30].tolist()
                            essential_cols = top_cols + text_columns + [target_column]
                            full_df = full_df[essential_cols]
                clear_memory()
        processed_file_path = analyzer.save_processed_dataset(full_df, os.path.basename(file.filename))
        analyzer.create_correlation_heatmap(full_df, target_column)
        if not analyzer.wordcloud_paths:
            analyzer.process_text_data(full_df, text_columns)
        if not analyzer.feature_importance_path and target_column in full_df.columns:
            analyzer.analyze_feature_importance(full_df, target_column)
        analyzer.create_sweetviz_report(full_df)
        analyzer.dataset_info['categorical_columns'] = len(text_columns)
        log_memory_usage("After processing")
        download_url = url_for('download_processed_data', filename=os.path.basename(processed_file_path))
        return render_template(
            'results.html',
            dataset_info=analyzer.dataset_info,
            heatmap_path=analyzer.heatmap_path,
            wordcloud_paths=analyzer.wordcloud_paths,
            sentiment_results=analyzer.sentiment_results,
            feature_importance_path=analyzer.feature_importance_path,
            sweetviz_report_path=analyzer.sweetviz_report_path,
            download_url=download_url
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        flash(f"Error processing data: {str(e)}")
        return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_processed_data(filename):
    try:
        return send_file(
            os.path.join('processed', filename),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'processed_{filename}'
        )
    except Exception as e:
        flash(f"Error downloading file: {str(e)}")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
