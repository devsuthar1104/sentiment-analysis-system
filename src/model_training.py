import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def initialize_model(self):
        """Initialize tokenizer and model"""
        print(f"🤖 Initializing model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model initialized successfully!")
    
    def prepare_datasets(self, X_train, y_train, X_val, y_val):
        """Prepare datasets for training"""
        print("📊 Preparing datasets...")
        train_dataset = SentimentDataset(X_train, y_train, self.tokenizer)
        val_dataset = SentimentDataset(X_val, y_val, self.tokenizer)
        print("✅ Datasets prepared!")
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset, val_dataset, output_dir="./models/sentiment_model"):
        """Train the sentiment analysis model"""
        print("🏋️ Starting model training...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Reduced for faster training
            per_device_train_batch_size=8,  # Reduced for lower memory usage
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        print("Training in progress...")
        self.trainer.train()
        
        # Save the model
        print("💾 Saving model...")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to {output_dir}")
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)
        
        results = []
        for i, text in enumerate(texts):
            confidence = float(predictions[i].max())
            sentiment = "positive" if predicted_classes[i] == 1 else "negative"
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence
            })
        
        return results

# Training script
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    print("🚀 Starting Complete Training Pipeline")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    
    # Step 2: Initialize and train model
    print("\n🤖 Initializing model...")
    analyzer = SentimentAnalyzer()
    analyzer.initialize_model()
    
    print("\n📚 Preparing datasets...")
    train_dataset, val_dataset = analyzer.prepare_datasets(X_train, y_train, X_val, y_val)
    
    print("\n🏋️ Training model...")
    analyzer.train_model(train_dataset, val_dataset)
    
    print("\n🎉 Training completed successfully!")
    
    # Quick test
    print("\n🧪 Quick test:")
    test_texts = ["I love this!", "This is terrible!"]
    results = analyzer.predict(test_texts)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print("-" * 30)