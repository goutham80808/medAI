# 🏥 Medic-AI: Disease Prediction & Medication Suggestion System

An intelligent machine learning application that forecasts possible diseases from reported symptoms and recommends appropriate drugs by analyzing patient feedback data.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.18+-red.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-orange.svg)

---

## 📌 Overview

`Medic-AI` integrates disease prediction with personalized drug suggestions using machine learning and NLP workflows. It leverages large-scale patient review data to extract insights and provide data-backed healthcare recommendations.

Inspired by methodologies in:

> *"An Intelligent Disease Prediction and Drug Recommendation Prototype by Using Multiple Approaches of Machine Learning Algorithms"*
> by Suvendu Kumar Nayak et al.

---

## 💡 Key Highlights

* Predicts likely diseases based on selected symptoms
* Suggests drugs with analysis of patient reviews and satisfaction levels
* Performs sentiment analysis on medication reviews
* User-friendly Streamlit web application
* Dynamic visualizations for better interpretation
* Supports multi-class disease classification

---

## 🗄 Data Source

The project utilizes the popular [UCI ML Drug Review dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018), originally curated for the KUC Hackathon Winter 2018.

### Dataset Snapshot

* Over **215,000** drug reviews from patients
* Associated conditions, star ratings, timestamps
* Rich text reviews detailing individual experiences

---

## ⚙ System Architecture

### Tech Stack

* **Frontend**: Streamlit (Python-based UI)
* **Core Language**: Python 3.8+
* **Machine Learning**: Scikit-learn
* **NLP & Text Processing**: NLTK, BeautifulSoup4
* **Data Wrangling**: Pandas, NumPy
* **Plots & Graphs**: Plotly, Matplotlib

### ML & NLP Pipeline

* Cleansing of reviews using NLTK
* Text vectorization via TF-IDF (supports n-grams)
* Classification using:

  * Passive Aggressive Classifier
  * Multinomial Naive Bayes
* Dedicated drug recommendation component based on review sentiments and frequency

---

## 🚀 Getting Started

### Requirements

* Python >= 3.8
* Git

### Quick Installation

```bash
git clone https://github.com/goutham80808/medAI.git
cd Medic-AI
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Insights

### Algorithms Implemented

* Bag of Words & TF-IDF feature extraction
* Passive Aggressive Classifier for rapid learning
* Naive Bayes for text sentiment and condition prediction

### Supported Health Conditions

* Hypertension
* Type 2 Diabetes
* Depression
* Asthma
* Pneumonia
* ADHD
* Acne
* Urinary Tract Infections
* Birth Control management
* Migraine
  ...and several more.

---

## 🗂 Directory Layout

```
Medic-AI/
├── app.py                  # Streamlit entry point
├── model/
│   ├── passmodel.pkl       # Serialized classifier
│   └── tfidfvectorizer.pkl # Saved vectorizer
├── data/
│   └── drug_reviews.csv    # Cleaned dataset
└── requirements.txt        # Python dependencies
└── Disease_condition_detection_drug_reviews.pynb
```

---

## 🚀 Usage Guide

1. Launch the app via Streamlit.
2. Choose symptoms from the interactive checklist.
3. Click on **Predict** to receive probable conditions.
4. Get recommended medications and view patient sentiment analysis plots.

---

---

## 📈 Performance Metrics

### 📊 Evaluated Models & Features

The models were evaluated using **accuracy** on the task of classifying patient conditions from drug reviews. Both models used **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.

| Model                         | Feature Extraction | Accuracy    |
| ----------------------------- | ------------------ | ----------- |
| Passive Aggressive Classifier | TF-IDF             | ≈ **93.9%** |
| Multinomial Naive Bayes       | TF-IDF             | ≈ **88.9%** |

### 📝 Dataset & Problem Scope

* The models tackled a **multi-class classification problem** across **10 medical conditions**:

  * Acne, ADHD, Asthma (acute), Birth Control, Depression, Diabetes Type 2, High Blood Pressure, Migraine, Pneumonia, Urinary Tract Infection
* Evaluations were based on a cleaned patient drug review dataset containing extensive textual feedback.

### ✅ Observations

* The **Passive Aggressive Classifier with TF-IDF** achieved the highest accuracy at approximately **94%**, making it the top performer for predicting patient conditions based on review text.
* The **Multinomial Naive Bayes model** also showed strong performance with an accuracy of around **89%**, serving as a solid baseline for medical text classification.

---

## ⚠ Disclaimer

This is an academic prototype meant for educational and demonstration purposes only. Always seek guidance from qualified healthcare professionals before making medical decisions.

---