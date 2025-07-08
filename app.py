# ==============================
# 📦 Import libraries
# ==============================
import os
import joblib
import pandas as pd
import re
import nltk
import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import plotly.express as px

# ==============================
# 📥 Download NLTK resources if missing
# ==============================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ==============================
# 🔥 Set up initial globals
# ==============================
predicted_cond = ""
top_drugs = []

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# ==============================
# ⚙️ Streamlit Page Config
# ==============================
st.set_page_config(
    page_title='DPDR',
    page_icon='👨‍⚕️',
    layout='wide'
)

# ==============================
# 🎨 Custom CSS
# ==============================
st.markdown("""
<style>
.stButton button {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-size: 18px;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton button:hover {
    background-color: #000000;
    color: #ffffff;
}
.stButton button:active {
    background-color: #89dceb;
}
.condition-card {
    padding: 20px;
    border-radius: 10px;
    background-color: #FFA500;
    color: #1e1e2e;
    margin: 10px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    font-weight: bold;
}
.drug-card-1, .drug-card-2, .drug-card-3, .drug-card-4 {
    padding: 15px;
    border-radius: 8px;
    color: #1e1e2e;
    margin: 10px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.drug-card-1 { background-color: #b4befe; }
.drug-card-2 { background-color: #a6e3a1; }
.drug-card-3 { background-color: #f38ba8; }
.drug-card-4 { background-color: #f2cdcd; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 📁 Sidebar Settings
# ==============================
st.sidebar.header("⚙️ Settings")
MODEL_PATH = st.sidebar.text_input("Model Path", value='model/passmodel.pkl')
TOKENIZER_PATH = st.sidebar.text_input("Tokenizer Path", value='model/tfidfvectorizer.pkl')
DATA_PATH = st.sidebar.text_input("Data Path", value='data/custom_dataset.csv')

# ==============================
# 🔍 Load resources safely
# ==============================
try:
    vectorizer = joblib.load(TOKENIZER_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model or vectorizer: {e}")
    st.stop()

try:
    df_data = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# ==============================
# 📝 Predefined symptoms list
# ==============================
symptoms = ["Acne", "Anxiety", "Depression", "High blood pressure", "Diabetes", "Migraine",
"Fever", "Fatigue", "Cough", "Shortness of breath", "Chest pain", "Headache", "Nausea",
"Joint pain", "Swelling", "Stress", "Mood swings", "Back pain", "Abdominal pain", 
"Weight gain", "Weight loss", "Blurred vision", "Increased thirst", "Increased hunger"]

# ==============================
# 🧹 Helper functions
# ==============================
def clean_text(raw_review):
    text = BeautifulSoup(raw_review, 'html.parser').get_text()
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    words = [lemmatizer.lemmatize(w) for w in text if w not in stop]
    return ' '.join(words)

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 90)].sort_values(
        by=['rating', 'usefulCount'], ascending=[False, False])
    drugs = df_top[df_top['condition'] == condition]['drugName'].unique()[:4]
    return list(drugs)

def predict_and_recommend(raw_text):
    global predicted_cond, top_drugs
    clean_input = clean_text(raw_text)
    tfidf_vect = vectorizer.transform([clean_input])
    prediction = model.predict(tfidf_vect)
    predicted_cond = prediction[0]
    top_drugs = top_drugs_extractor(predicted_cond, df_data)

# ==============================
# 🎯 Main UI
# ==============================
st.title("💉 Disease Prediction and Drug Recommendation")
st.markdown("---")
st.header("📝 Enter Patient Symptoms")

# Input selection
input_method = st.radio("Choose input method:", ["Select from predefined list", "Type your own text"])
raw_text = ""

if input_method == "Select from predefined list":
    selected_symptoms = st.multiselect("Choose symptoms:", symptoms)
    raw_text = ", ".join(selected_symptoms)
else:
    raw_text = st.text_area("Describe the patient's symptoms:", height=100)

st.markdown(f"**Input Text:** {raw_text}")

# ==============================
# 🚀 Predict
# ==============================
if st.button("🔍 Predict"):
    if not raw_text.strip():
        st.warning("⚠️ Please enter or select symptoms.")
    else:
        with st.spinner("🧠 Analyzing..."):
            predict_and_recommend(raw_text)
        
        st.markdown("---")
        st.markdown("### 🎯 Condition Predicted")
        st.markdown(f"<div class='condition-card'><h3>{predicted_cond}</h3></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💊 Top Recommended Drugs")
        for i, drug in enumerate(top_drugs):
            st.markdown(f"<div class='drug-card-{i+1}'><h4>{i+1}. {drug}</h4></div>", unsafe_allow_html=True)

        # Plotly visualization
        if top_drugs:
            st.markdown("---")
            st.markdown("### 📊 Drug Recommendations Visualization")
            df_plot = pd.DataFrame({"Drug": top_drugs[::-1], "Rank": list(range(len(top_drugs),0,-1))})
            color_map = {}
            if len(top_drugs) > 0: color_map[top_drugs[0]] = "#b4befe"
            if len(top_drugs) > 1: color_map[top_drugs[1]] = "#a6e3a1"
            if len(top_drugs) > 2: color_map[top_drugs[2]] = "#f38ba8"
            if len(top_drugs) > 3: color_map[top_drugs[3]] = "#f2cdcd"

            fig = px.bar(df_plot, x="Rank", y="Drug", orientation='h', text_auto=True, 
                         color="Drug", color_discrete_map=color_map,
                         title="Top Recommended Drugs")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

# ==============================
# ⚠️ Disclaimer
# ==============================
st.markdown("---")
st.warning("""
⚠️ **Disclaimer**  
This is a prototype. It is not a substitute for professional medical advice, diagnosis or treatment.
Always seek advice from qualified healthcare providers.
""")
st.markdown("---")
with st.expander("ℹ️ About This App"):
    st.write("""
This app predicts diseases from symptoms and recommends drugs based on user reviews and ratings.
- Built using Python, Streamlit, NLTK, Scikit-learn, and Plotly.
    """)

