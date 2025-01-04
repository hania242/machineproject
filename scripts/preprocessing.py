# preprocessing.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def preprocess_data(input_file, cleansed_csv, output_file, raw_text_file):
   
    dataset = pd.read_csv(input_file)

    
    dataset = dataset.dropna(subset=['label'])

    X_raw = dataset['text']

   
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  
        return text

    dataset['clean_text'] = dataset['text'].apply(clean_text)

   
    dataset.to_csv(cleansed_csv, index=False)
    print(f"Cleansed dataset saved to {cleansed_csv}")

    X = dataset['clean_text']
    y = dataset['label']

    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    
    with open(output_file, 'wb') as f:
        pickle.dump((X_tfidf, y, vectorizer), f)

    with open(raw_text_file, 'wb') as f:
        pickle.dump(X_raw, f)

    print(f"Preprocessed data saved to {output_file}")
    print(f"Raw text data saved to {raw_text_file}")
