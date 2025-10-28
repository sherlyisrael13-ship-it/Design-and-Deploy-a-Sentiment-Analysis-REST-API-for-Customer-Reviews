import os
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 

class Review(BaseModel):
    """Schema for the incoming request body."""
    review_text: str

class SentimentResponse(BaseModel):
    """Schema for the outgoing JSON response, including metadata (M4)."""
    review_text: str
    sentiment: str
    score: float
    request_id: str
    model_version: str
    timestamp: str

app = FastAPI(title="Sentiment Analysis API", version="1.0")

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

def analyze_sentiment_logic(text: str) -> tuple[str, float]:
    """
    Placeholder for calling your Vertex AI model or running local analysis.
    For demonstration, we use simple keyword analysis.
    """
    text_lower = text.lower()

    positive_keywords = ["great", "love", "fantastic", "excellent", "best"]
    negative_keywords = ["bad", "hate", "terrible", "slow", "unreliable"]
    
    pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    score = pos_count - neg_count
    
    if score > 0:
        sentiment = "POSITIVE"
    elif score < 0:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
        
    return sentiment, score

@app.get("/")
def home():
    """Simple check to ensure the service is running."""
    return {"status": "ok", "service": "Sentiment Analysis API is ready."}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(review: Review):
    """Receives text and returns sentiment analysis and M4 metadata."""

    sentiment, score = analyze_sentiment_logic(review.review_text)

    response_metadata = SentimentResponse(
        review_text=review.review_text,
        sentiment=sentiment,
        score=float(score),
        request_id=f"req-{os.getpid()}-{int(time.time())}", 
        model_version="v1.0.0-vertex-ai-base",
        timestamp=datetime.now().isoformat(),
    )
    
    return response_metadata
