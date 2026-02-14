"""
FastAPI application for spam detection
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

# Initialize FastAPI app
app = FastAPI(
    title="Email Spam Detection API",
    description="Detect if an email is spam or not using Machine Learning",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = '../model/spam_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Define request model
class EmailText(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You have won a $1000 prize. Click here to claim now!"
            }
        }


# Define response model
class SpamPrediction(BaseModel):
    text: str
    prediction: str
    is_spam: bool
    confidence: float


# Root endpoint
@app.get("/")
async def root():
    """
    Welcome endpoint
    """
    return {
        "message": "Welcome to Email Spam Detection API",
        "endpoints": {
            "/predict": "POST - Predict if email is spam",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Check if API and model are working
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict", response_model=SpamPrediction)
async def predict_spam(email: EmailText):
    """
    Predict if email text is spam or not

    - **text**: Email text to classify

    Returns:
    - **prediction**: "Spam" or "Ham" (Not Spam)
    - **is_spam**: Boolean value
    - **confidence**: Prediction confidence (0-1)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Get prediction
        prediction = model.predict([email.text])[0]

        # Get probability scores
        probabilities = model.predict_proba([email.text])[0]
        confidence = float(max(probabilities))

        # Prepare response
        is_spam = bool(prediction == 1)
        prediction_label = "Spam" if is_spam else "Ham (Not Spam)"

        return SpamPrediction(
            text=email.text,
            prediction=prediction_label,
            is_spam=is_spam,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch prediction endpoint
@app.post("/predict-batch")
async def predict_batch(emails: list[EmailText]):
    """
    Predict spam for multiple emails at once
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    if len(emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 emails per batch")

    try:
        texts = [email.text for email in emails]
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)

        results = []
        for i, email in enumerate(emails):
            is_spam = bool(predictions[i] == 1)
            confidence = float(max(probabilities[i]))

            results.append({
                "text": email.text[:100] + "..." if len(email.text) > 100 else email.text,
                "prediction": "Spam" if is_spam else "Ham",
                "is_spam": is_spam,
                "confidence": confidence
            })

        return {"results": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)