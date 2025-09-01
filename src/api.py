from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import time
from datetime import datetime

app = FastAPI(
    title="Sentiment Analysis API",
    description="Multi-language sentiment analysis with real-time processing",
    version="1.0.0"
)

# Load model globally
MODEL_PATH = "./models/sentiment_model"
tokenizer = None
model = None

def load_model():
    """Load the trained model"""
    global tokenizer, model
    try:
        print("🤖 Loading trained model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

class TextInput(BaseModel):
    text: str
    language: Optional[str] = "en"

class BatchTextInput(BaseModel):
    texts: List[str]
    language: Optional[str] = "en"

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processing_time: float
    language: str

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time: float
    batch_size: int

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎭 Sentiment Analysis API is running!",
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment for a single text"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Tokenize input
        inputs = tokenizer(
            input_data.text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
            confidence = float(predictions.max())
        
        sentiment = "positive" if predicted_class.item() == 1 else "negative"
        processing_time = time.time() - start_time
        
        return SentimentResponse(
            text=input_data.text,
            sentiment=sentiment,
            confidence=confidence,
            processing_time=processing_time,
            language=input_data.language
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(input_data: BatchTextInput):
    """Predict sentiment for multiple texts"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    results = []
    
    try:
        for text in input_data.texts:
            text_start_time = time.time()
            
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1)
                confidence = float(predictions.max())
            
            sentiment = "positive" if predicted_class.item() == 1 else "negative"
            text_processing_time = time.time() - text_start_time
            
            results.append(SentimentResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                processing_time=text_processing_time,
                language=input_data.language
            ))
        
        total_processing_time = time.time() - start_time
        
        return BatchSentimentResponse(
            results=results,
            total_processing_time=total_processing_time,
            batch_size=len(input_data.texts)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_PATH,
        "model_type": type(model).__name__,
        "tokenizer_vocab_size": len(tokenizer.vocab) if tokenizer else 0,
        "supported_languages": ["en"],
        "max_sequence_length": 512
    }

if __name__ == "__main__":
    print("🚀 Starting Sentiment Analysis API...")
    uvicorn.run(app, host="127.0.0.1", port=8000)