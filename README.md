# 📱 SMS Spam Detection System

🚨 **Detect fraudulent SMS messages in real-time** using **Machine Learning** and **Deep Learning**.  
This project compares classical ML models (Logistic Regression, SVM, Random Forest, Naive Bayes) with advanced Deep Learning (BiLSTM) to classify SMS messages as **Ham (legit)** or **Spam (fraudulent)**.

Built with **Python, Scikit-learn, TensorFlow, and Streamlit**.

---

## ✨ Features

* 📊 **Model Comparison**: Logistic Regression, SVM, Random Forest, Naive Bayes, BiLSTM  
* ⚡ **Real-time Prediction App** (Streamlit)  
* 🧠 **Deep Learning with BiLSTM** (captures sequential word patterns)  
* 🔍 **Explainable Spam Indicators** (urgency words, prize mentions, ALL CAPS, etc.)  
* 🤝 **Ensemble Predictions** (combine ML + DL for better accuracy)  
* 📈 **Performance Metrics**: Accuracy, Precision, Recall, F1-score  

---

## 🚀 Demo (App UI)

### Spam Example:

CONGRATULATIONS! You've won $1000! Call now to claim your prize!


➡️ Prediction: **🚨 SPAM (98% confidence)**

### Ham Example:



Hey, are we still meeting for lunch tomorrow?


➡️ Prediction: **✅ HAM (99% confidence)**


---


## 🛠️ Tech Stack

* **Programming Language**: Python  
* **Libraries**: Numpy, Pandas, Scikit-learn, TensorFlow, Streamlit  
* **Visualization**: Matplotlib, Seaborn  
* **Deployment**: Streamlit (local or cloud)  

---

## 📊 Model Performance

| Model               | Accuracy  | Precision | Recall | F1-score  |
| ------------------- | --------- | --------- | ------ | --------- |
| Logistic Regression | 95.2%     | 96.5%     | 64.1%  | 76.9%     |
| SVM (Linear)        | 97.4%     | 97.2%     | 81.2%  | 88.5%     |
| Random Forest       | 96.7%     | 98.0%     | 75.0%  | 84.9%     |
| **BiLSTM**          | **98.4%** | 97.5%     | 89.8%  | **93.5%** |

✅ **BiLSTM is the best performing model.**

---

## ⚙️ Installation & Usage

1. Clone the repo:

```bash
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3.Run Streamlit app:
``` 
streamlit run app.py
```

## 🧠 Methodology

* Data Preprocessing: clean text (lowercasing, removing punctuation, stopwords)

* Feature Engineering: TF-IDF for ML models, TextVectorization + Embeddings for BiLSTM

* Model Training: Logistic Regression, SVM, Random Forest, Naive Bayes, BiLSTM

* Evaluation: Accuracy, Precision, Recall, F1-score

* Deployment: Streamlit app for real-time predictions

## 🔮 Future Improvements


* 📲 Extend system to WhatsApp/Email Spam Detection

* 🔍 Add SHAP/LIME explainability for model decisions

* ⚡ Optimize BiLSTM with pre-trained embeddings (GloVe/FastText)

  ---

## 🌐 Live Demo

🚀 Try out the app here: [SMS Spam Detection App](https://your-streamlit-link.streamlit.app)  


---

