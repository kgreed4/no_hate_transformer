from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=150)

    def preprocess(self, data_series):
        """
        Preprocess the text data by removing special characters and digits, tokenizing, and stemming and lemmatizing the words.
        """
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = text.split()
            tokens = [self.stemmer.stem(word) for word in tokens]
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(tokens)
        
        return data_series.apply(clean_text)

    def extract_features(self, preprocessed_data):
        """
        Extract features from the preprocessed data using TF-IDF.
        """
        tfidf_features = self.vectorizer.fit_transform(preprocessed_data)
        features_df = pd.DataFrame(tfidf_features.toarray())
        return features_df

    def split_data(self, data_features, labels, test_size=0.2, val_size=0.25):
        """
        Splits data into training, validation, and testing sets.
        """
        X_temp, X_test, y_temp, y_test = train_test_split(data_features, labels, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def process_and_split(self, data_series, labels, test_size=0.2, val_size=0.25):
        """
        A function that calls the other three: preprocess, extract_features, and split_data.
        """
        preprocessed_data = self.preprocess(data_series)
        features = self.extract_features(preprocessed_data)
        return self.split_data(features, labels, test_size, val_size)