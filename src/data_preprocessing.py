import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.data = None
        
    def load_data(self):
        """Load sample sentiment data"""
        print("Loading sample sentiment data...")
        
        # Sample positive reviews
        positive_reviews = [
            "I absolutely love this movie! Amazing acting and great story.",
            "This product is fantastic! Highly recommended to everyone.",
            "Best purchase I've ever made. Outstanding quality and service.",
            "Incredible experience! Will definitely come back for more.",
            "Perfect service and excellent customer support team.",
            "Amazing quality! Exceeded all my expectations completely.",
            "Wonderful experience from start to finish. Highly satisfied.",
            "Excellent product! Works perfectly and great value for money.",
            "Outstanding service! Quick delivery and perfect packaging.",
            "Love it! Best decision I made this year."
        ] * 200  # 2000 positive samples
        
        # Sample negative reviews
        negative_reviews = [
            "Terrible movie. Complete waste of time and money.",
            "Poor quality product. Very disappointed with purchase.",
            "Worst experience ever. Would not recommend to anyone.",
            "Bad service and extremely rude staff members.",
            "Complete waste of money. Awful quality and design.",
            "Horrible experience! Never buying from here again.",
            "Disappointed with quality. Not worth the price at all.",
            "Poor customer service. No response to complaints.",
            "Defective product. Stopped working after two days.",
            "Overpriced and low quality. Very unsatisfied customer."
        ] * 200  # 2000 negative samples
        
        # Create DataFrame
        texts = positive_reviews + negative_reviews
        labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
        
        self.data = pd.DataFrame({
            'text': texts,
            'label': labels,
            'language': ['en'] * len(texts)
        })
        
        # Shuffle the data
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Positive samples: {len(self.data[self.data['label'] == 1])}")
        print(f"Negative samples: {len(self.data[self.data['label'] == 0])}")
        return self.data
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_data(self):
        """Preprocess the entire dataset"""
        print("Preprocessing data...")
        
        # Clean text
        self.data['cleaned_text'] = self.data['text'].apply(self.clean_text)
        
        # Remove empty texts
        self.data = self.data[self.data['cleaned_text'].str.len() > 10]
        
        print(f"After cleaning: {len(self.data)} samples")
        return self.data
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        X = self.data['cleaned_text']
        y = self.data['label']
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# Usage example
if __name__ == "__main__":
    print("🚀 Starting Data Preprocessing...")
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    
    print("\n✅ Data Preprocessing Results:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    print("\n🎉 Data preprocessing completed successfully!")