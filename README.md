# Email Spam Detection API - MLOps LAB2

## Project Description
A FastAPI-based REST API that uses Machine Learning to detect spam emails. The model is trained on a real-world spam email dataset from Kaggle and achieves **97.47% accuracy**.

## My Modifications
Instead of the standard Iris classification example, I created:
- **Email Spam Detection API** using a real Kaggle dataset
- **5,728 email samples** (1,368 spam, 4,360 ham)
- **Naive Bayes classifier** with TF-IDF vectorization
- **Batch prediction endpoint** for multiple emails
- **Interactive API documentation** with FastAPI
- **97.47% accuracy** on test data

## Dataset
- **Source**: [Kaggle Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)
- **Total Samples**: 5,728 emails
- **Spam**: 1,368 (23.9%)
- **Ham (Not Spam)**: 4,360 (76.1%)

## Technologies Used
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI web server
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **Pydantic**: Data validation
- **Naive Bayes**: Classification algorithm
- **TF-IDF**: Text vectorization

## Model Performance
```
Accuracy: 97.47%

Classification Report:
              precision    recall  f1-score
         Ham       0.97      0.99      0.98
        Spam       0.98      0.92      0.95
```


The API will be available at: `http://127.0.0.1:8000`

## API Endpoints

### 1. Root Endpoint
- **URL**: `GET /`
- **Description**: Welcome message and endpoint list

### 2. Health Check
- **URL**: `GET /health`
- **Description**: Check if API and model are working

### 3. Predict Single Email
- **URL**: `POST /predict`
- **Request Body**:
```json
{
  "text": "Your email text here"
}
```
- **Response**:
```json
{
  "text": "Your email text here",
  "prediction": "Spam" or "Ham (Not Spam)",
  "is_spam": true or false,
  "confidence": 0.95
}
```

### 4. Batch Prediction
- **URL**: `POST /predict-batch`
- **Description**: Predict spam for up to 100 emails at once

## Testing the API

### Using Interactive Documentation:
1. Go to `http://127.0.0.1:8000/docs`
2. Click on any endpoint
3. Click "Try it out"
4. Enter your test data
5. Click "Execute"

### Example Test Cases:

**Spam Email:**
```json
{
  "text": "Congratulations! You won $1000000! Click here to claim now!"
}
```

**Normal Email:**
```json
{
  "text": "Hi, can we schedule a meeting tomorrow at 3pm?"
}
```

## Screenshots
API testing screenshots are available in the `output/` folder showing:
- API documentation interface
- Spam email detection
- Normal email (ham) detection

## Features
- Real-world spam email dataset  
- 97.47% accuracy  
- FastAPI with automatic API documentation  
- Single and batch prediction endpoints  
-  Input validation with Pydantic  
-  Error handling  
-  Health check endpoint  
-  Interactive testing interface  

## Author
Afrah Fathima

## Course
MLOps (IE-7374) - LAB2

