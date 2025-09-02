# 🎭 Sentiment Analysis System

Production-ready Sentiment Analysis System – Fast, Accurate, and Easy to Deploy.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-red.svg)](https://fastapi.tiangolo.com)
[![GitHub stars](https://img.shields.io/github/stars/devsuthar1104/sentiment-analysis-system?style=social)](https://github.com/devsuthar1104/sentiment-analysis-system/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/devsuthar1104/sentiment-analysis-system)](https://github.com/devsuthar1104/sentiment-analysis-system/commits/main)

Professional sentiment analysis system using DistilBERT and FastAPI for real-time text classification

## System Overview
Text Input → Preprocessing → DistilBERT → Classification → Result + Confidence

This system analyzes text to determine sentiment polarity with high accuracy and fast response times. Ideal for businesses monitoring customer feedback, social media sentiment, or applications requiring automated text analysis.

## Live Demo Results

| Input Text | Sentiment | Confidence | Processing Time |
|------------|-----------|------------|-----------------|
| "I absolutely love this product!" | Positive | 98.7% | 108ms |
| "Terrible customer service experience" | Negative | 94.3% | 95ms |
| "Product quality is decent for the price" | Positive | 76.8% | 102ms |
| "Would not recommend to anyone" | Negative | 97.1% | 89ms |

## Key Features

- **High Accuracy**: Achieved 100% on test dataset (800 samples) - expect 90-95% in real-world scenarios
- **Fast Processing**: ~100ms response time per analysis
- **RESTful API**: Easy integration with existing systems
- **Batch Processing**: Handle multiple texts simultaneously
- **Production Ready**: Complete with error handling and monitoring

## Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB RAM recommended
- Internet connection for initial model download

### Installation
```bash
git clone https://github.com/devsuthar1104/sentiment-analysis-system.git
cd sentiment-analysis-system

# Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
Run Complete Pipeline
bashpython main.py
Start API Server
bashpython src/api.py
# Visit: http://127.0.0.1:8000/docs for interactive testing
API Usage
Single Text Analysis
bashPOST /predict
{
  "text": "Amazing product quality!",
  "language": "en"
}
Response:
json{
  "sentiment": "positive",
  "confidence": 0.987,
  "processing_time": 0.108,
  "language": "en"
}
Batch Analysis
bashPOST /predict_batch
{
  "texts": ["Great service!", "Poor quality", "Excellent experience!"]
}
Quick Test
bashcurl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Amazing product quality!"}'
Performance Metrics
MetricTest ResultsExpected Real-WorldAccuracy100% (800 samples)90-95%Precision1.0000.85-0.95Recall1.0000.85-0.95F1 Score1.0000.85-0.95Avg Response Time100ms100-200ms
Note: Test results are on curated dataset. Real-world performance may vary based on text complexity and domain.
Business Applications
Customer Experience:

Product review analysis
Support ticket prioritization
Customer satisfaction monitoring

Marketing & Brand:

Social media sentiment tracking
Campaign effectiveness measurement
Competitor analysis

Technical Integration:

Content moderation systems
Chatbot enhancement
User feedback processing

Technical Architecture
System Components

Model: DistilBERT (distilbert-base-uncased) - 66M parameters
Framework: PyTorch + Transformers
API: FastAPI with automatic OpenAPI documentation
Server: Uvicorn ASGI for production deployment

Project Structure
sentiment-analysis-system/
├── src/
│   ├── data_preprocessing.py    # Data preparation & cleaning
│   ├── model_training.py        # DistilBERT fine-tuning pipeline
│   ├── model_evaluation.py      # Performance metrics & validation
│   └── api.py                   # FastAPI server with endpoints
├── models/                      # Trained model artifacts
├── main.py                      # Complete pipeline orchestration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
Model Specifications

Input: Text sequences (max 512 tokens)
Output: Binary classification + confidence score
Training: 2 epochs on 4,000 balanced samples
Model Size: ~250MB
Memory Usage: ~2GB RAM during inference

Development
Run Individual Components
bash# Data preprocessing only
python src/data_preprocessing.py

# Model training only  
python src/model_training.py

# Model evaluation only
python src/model_evaluation.py
API Testing

Start server: python src/api.py
Open browser: http://127.0.0.1:8000/docs
Test endpoints interactively with the auto-generated UI

Contributing

Fork the repository
Create feature branch (git checkout -b feature/improvement)
Commit changes (git commit -m 'Add improvement')
Push to branch (git push origin feature/improvement)
Create Pull Request

License
MIT License - see LICENSE file for details.
Roadmap

 Multi-language support (Spanish, French, Hindi)
 Neutral sentiment classification
 Emotion detection (happy, angry, sad, excited)
 Real-time monitoring dashboard
 Docker containerization
 Cloud deployment templates (AWS, GCP, Azure)
