import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

class ModelEvaluator:
    def __init__(self, model_path="./models/sentiment_model"):
        self.model_path = model_path
        print("🔍 Loading model for evaluation...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
    
    def predict_single(self, text):
        """Predict sentiment for single text"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1)
            confidence = float(probs.max())
        
        return predicted_class.item(), confidence
    
    def evaluate_model(self, X_test, y_test):
        """Simple model evaluation"""
        print(f"\n📊 Evaluating model on {len(X_test)} test samples...")
        
        predictions = []
        confidences = []
        
        # Process in small batches to show progress
        batch_size = 50
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            print(f"   Processing batch {i//batch_size + 1}: samples {i+1}-{batch_end}")
            
            for j in range(i, batch_end):
                pred, conf = self.predict_single(X_test.iloc[j])
                predictions.append(pred)
                confidences.append(conf)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        print("\n📈 EVALUATION RESULTS:")
        print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 F1 Score: {f1:.4f}")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   🎯 Recall: {recall:.4f}")
        print(f"   🎯 Average Confidence: {np.mean(confidences):.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'average_confidence': np.mean(confidences)
        }

# Test the evaluator
if __name__ == "__main__":
    print("🚀 Starting Model Evaluation")
    print("=" * 40)
    
    try:
        # Check if model exists
        model_path = "./models/sentiment_model"
        if not os.path.exists(model_path):
            print("❌ Model not found! Please run training first.")
            exit()
        
        # Load test data
        print("\n📊 Loading test data...")
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data()
        processed_data = preprocessor.preprocess_data()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
        
        print(f"✅ Test data loaded: {len(X_test)} samples")
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_model(X_test, y_test)
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()