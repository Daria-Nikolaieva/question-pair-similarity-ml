# 🔍 Duplicate Questions Detection

## 1. Project Overview

Duplicate Questions Detection is a machine learning project that aims to identify whether a pair of questions has the same meaning. The goal is to detect semantically equivalent questions, even when they are phrased differently.

## 2. Business Problem and Objective

On many platforms, such as FAQ systems, customer support services, and Q&A websites, users often ask the same question using different wording. This results in duplicated content, increased workload for moderators, and a poorer user experience.

The objective of this project is to build a machine learning model that can automatically identify duplicate questions with high accuracy and robust performance.

## 3. Dataset

- **Source:** Quora Question Pairs  
  https://www.kaggle.com/c/quora-question-pairs

- **Size:** ~400,000 question pairs

- **Features:**
  - `question1` — first question
  - `question2` — second question
  - `is_duplicate` — target variable (`1` for duplicate questions, `0` otherwise)

The dataset contains a wide variety of question formulations, including synonyms and paraphrases, making duplicate detection a challenging semantic classification task.

## 4. Evaluation Strategy

The primary evaluation metric is **Log Loss (Cross-Entropy Loss)** because:

- the task is a binary classification problem with probabilistic outputs;
- it evaluates both prediction accuracy and confidence;
- it is well suited for comparing individual models and ensemble approaches.

Additional evaluation methods include:

- Confusion Matrix
- Precision / Recall
- Decision threshold tuning

## 5. Solution Approach and Tech Stack

Several approaches were explored throughout the project.

### 🔹 Feature Engineering

- Lexical and string similarity metrics (Jaccard, Levenshtein)
- TF-IDF representations with distance-based features

### 🔹 Semantic Features

- SBERT embeddings
- Cosine similarity, L2 distance, and dot product

### 🔹 Models

- Logistic Regression (baseline)
- Random Forest
- LightGBM
- MLP with SBERT embeddings
- Weighted Ensemble (LightGBM + MLP)

### 🛠 Technologies

- Python, NumPy, Pandas
- Scikit-learn
- LightGBM
- PyTorch
- Sentence-BERT
- FastAPI (deployment)

## 6. Results
### 📊 Model Comparison

| Model | Features | Train Log Loss | Val Log Loss |
|------|--------|----------------|--------------|
| Logistic Regression | Handcrafted features | 0.5639 | 0.5663 | 
| Random Forest | Handcrafted features | 0.1569 | 0.5519 |
| XGBoost | Handcrafted features | 0.4868 | 0.4958 | 
| LightGBM | Handcrafted features | 0.4824 | 0.4938 | 
| Logistic Regression | TF-IDF | 0.4766 | 0.5242 |
| Linear SVM | TF-IDF | 0.4494 | 0.5284 | 
| SGDClassifier | TF-IDF | 0.5480 | 0.5542 |
| Logistic Regression | TF-IDF + handcrafted | 0.3888 | 0.4327 | 
| Linear SVM | TF-IDF + handcrafted | 0.4648 | 0.5037 |
| LightGBM | TF-IDF + handcrafted | 0.3957 | 0.4102 | 
| SBERT + Logistic Regression | SBERT Cosine | 0.4228 | 0.4221 |
| LightGBM SBERT | Sentence embeddings | 0.2879 | 0.3217 | 
| MLP + SBERT | Sentence embeddings | 0.1248| 0.2822 | 
| LightGBM | TF-IDF + handcrafted + SBERT Cosine | 0.2962 | 0.3087 | 
| Ensemble | TF-IDF + handcrafted + SBERT Cosine + Sentence embeddings| 0.2683 | 0.2683 | 

The ensemble model reduced the number of false positives and achieved a better balance between different types of classification errors.

## 7. Conclusions

- Semantic embeddings (SBERT) significantly outperform classical approaches.
- LightGBM performs well with handcrafted features.
- MLP can effectively capture deeper semantic patterns from SBERT embeddings.
- Model ensembling improves prediction stability and overall performance.

## 8. How to Run the Project (Installation & Usage)

🔹 **Clone the repository**

```bash
git clone https://github.com/Daria-Nikolaieva/question-pair-similarity-ml.git
cd duplicate-questions
```

🔹 **Install dependencies**

```bash
pip install -r requirements.txt
```

🔹 **Run the API**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

🔹 **Send a request to the API**

Endpoint:

```
POST /predict
```

Example request:

```json
{
  "question1": "How can I learn machine learning?",
  "question2": "What is the best way to study ML?"
}
```

## 9. Requirements (`requirements.txt`)

```text
fastapi==0.127.0
uvicorn==0.40.0
torch==2.9.0
sentence-transformers==5.2.0
huggingface-hub==0.27.0
numpy==1.26.2
pandas==2.1.1
scikit-learn==1.7.2
scipy==1.11.2
lightgbm==4.0.0
joblib==1.3.2
pydantic==2.10.4
nltk==3.8.1
rapidfuzz==3.14.1
protobuf==5.29.3
```
