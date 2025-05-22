# main.py — Backend API for Review Summarization and Fake Review Detection

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, BertForSequenceClassification
import torch

# === Data and Model Setup ===
file_path = "fake reviews dataset.csv"  # Ensure this file is in the same directory

# Load and preprocess data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'category' not in df.columns or 'text_' not in df.columns:
        raise ValueError("Dataset must contain 'category' and 'text_' columns.")
    df = df[df['text_'].notnull()]
    df['text'] = df['text_'].apply(preprocess_text)
    return df

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_summary_pipeline():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# Load classifier pipeline
def load_classification_model():
    model_name = "bert_fake_review_model"  # You can replace with your own model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_fake_review(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label = "Fake" if predicted_class == 1 else "Real"
    return label

# Load everything once
print("Loading models and data...")
df = load_and_clean_data(file_path)
summarizer = load_summary_pipeline()
bert_tokenizer, bert_model = load_classification_model()
print("✅ All models and data loaded")

# === Load Semantic Search Model ===
from sentence_transformers import SentenceTransformer, util
import pickle

print("Loading semantic search model and embeddings...")
semantic_model = SentenceTransformer("saved_model")
semantic_embeddings = torch.load("saved_model/embeddings.pt", map_location=torch.device('cpu'))
with open("saved_model/texts.pkl", "rb") as f:
    semantic_texts = pickle.load(f)
print("✅ Semantic search engine ready")

# === Load Explainable Summarization Model ===
print("Loading explainable summarization model...")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
explain_model_name = "google/flan-t5-base"  # You can change to t5-small or bart-base
explain_tokenizer = AutoTokenizer.from_pretrained(explain_model_name)
explain_model = AutoModelForSeq2SeqLM.from_pretrained(explain_model_name)
explain_pipe = pipeline("text2text-generation", model=explain_model, tokenizer=explain_tokenizer)
print("✅ Explanation model ready")

# === FastAPI Setup ===
app = FastAPI(title="Fake Review Detection & Summarization API")

class SummaryRequest(BaseModel):
    category: str

class ReviewRequest(BaseModel):
    review_text: str

"""@app.post("/summarize")
def summarize_reviews(req: SummaryRequest):
    category = req.category
    product_reviews = df[df['category'] == category]['text'].tolist()

    if not product_reviews:
        raise HTTPException(status_code=404, detail="No reviews found for this category.")

    full_text = " ".join(product_reviews[:30])
    summary = summarizer(full_text, max_length=120, min_length=30, do_sample=False)
    return {"category": category, "summary": summary[0]['summary_text']}"""

@app.post("/predict-fake-review")
def classify_review(req: ReviewRequest):
    review = preprocess_text(req.review_text)
    label = predict_fake_review(review, bert_tokenizer, bert_model)
    return {"review": req.review_text, "prediction": label}

@app.post("/semantic-search")
def semantic_search(req: ReviewRequest):
    query_embedding = semantic_model.encode(req.review_text, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, semantic_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({"score": float(score), "review": semantic_texts[idx]})

    return {"query": req.review_text, "results": results}

@app.post("/explain-review")
def explain_review(req: ReviewRequest):
    prompt = f"Explain why the following review might be fake: {req.review_text}"
    explanation = explain_pipe(prompt, max_length=64, clean_up_tokenization_spaces=True)
    return {"review": req.review_text, "explanation": explanation[0]['generated_text']}

# Run using: uvicorn main:app --reload
