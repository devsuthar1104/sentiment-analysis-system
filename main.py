import os
import sys
from src.data_preprocessing import DataPreprocessor
from src.model_training import SentimentAnalyzer, SentimentDataset
from src.model_evaluation import ModelEvaluator

def main():
    """Complete pipeline execution"""
    print("🚀 SENTIMENT ANALYSIS PROJECT - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n📊 STEP 1: DATA PREPROCESSING")
    print("-" * 30)
    preprocessor = DataPreprocessor()
    
    # Load data
    data = preprocessor.load_data()
    
    # Preprocess data
    processed_data = preprocessor.preprocess_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    
    print(f"✅ Data preprocessing completed!")
    print(f"   📈 Train samples: {len(X_train)}")
    print(f"   📈 Validation samples: {len(X_val)}")
    print(f"   📈 Test samples: {len(X_test)}")
    
    # Step 2: Model Training (Skip if model exists)
    model_path = "./models/sentiment_model"
    if os.path.exists(model_path):
        print(f"\n🤖 STEP 2: MODEL ALREADY TRAINED")
        print("-" * 30)
        print(f"✅ Model found at {model_path}")
        print("   Skipping training (already completed)")
    else:
        print(f"\n🤖 STEP 2: MODEL TRAINING")
        print("-" * 30)
        analyzer = SentimentAnalyzer()
        analyzer.initialize_model()
        
        # Prepare datasets
        train_dataset, val_dataset = analyzer.prepare_datasets(X_train, y_train, X_val, y_val)
        
        # Train model
        analyzer.train_model(train_dataset, val_dataset)
        print("✅ Model training completed!")
    
    # Step 3: Model Evaluation
    print(f"\n📈 STEP 3: MODEL EVALUATION")
    print("-" * 30)
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(X_test, y_test)
    
    print("✅ Model evaluation completed!")
    
    # Step 4: Quick Demo
    print(f"\n🧪 STEP 4: QUICK DEMO")
    print("-" * 30)
    
    # Create a simple predictor for demo
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    demo_texts = [
        "I absolutely love this amazing product!",
        "This is the worst thing I've ever bought.",
        "Pretty good quality for the price.",
        "Terrible customer service experience."
    ]
    
    print("🔮 Demo Predictions:")
    for text in demo_texts:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1)
            confidence = float(probs.max())
        
        sentiment = "positive" if predicted_class.item() == 1 else "negative"
        print(f"   📝 Text: '{text[:50]}...'")
        print(f"   🎭 Sentiment: {sentiment.upper()} (Confidence: {confidence:.3f})")
        print()
    
    # Step 5: API Instructions
    print(f"\n🌐 STEP 5: API USAGE")
    print("-" * 30)
    print("To start the API server:")
    print("   python src/api.py")
    print()
    print("API Endpoints:")
    print("   🏠 Home: http://127.0.0.1:8000/")
    print("   📚 Docs: http://127.0.0.1:8000/docs")
    print("   🔍 Health: http://127.0.0.1:8000/health")
    print("   🎭 Predict: http://127.0.0.1:8000/predict")
    
    print(f"\n🎯 PROJECT SUMMARY")
    print("=" * 60)
    print(f"✅ Data Processing: {len(processed_data)} samples processed")
    print(f"✅ Model Training: DistilBERT fine-tuned successfully")
    print(f"✅ Model Performance: {results['accuracy']*100:.1f}% accuracy")
    print(f"✅ API Development: FastAPI server ready")
    print(f"✅ GitHub Ready: Complete project structure")
    
    print(f"\n🚀 YOUR SENTIMENT ANALYSIS SYSTEM IS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    import torch
    main()