# 📩 SMS Spam Classifier

A simple and effective machine learning app that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and the Naive Bayes algorithm. Built with ❤️ using **Streamlit**, **scikit-learn**, and **NLTK**.

---

## 🚀 Live Demo

👉 [Click here to open the app](https://smsspamclassifier-edwnnvw2ovwec7d8vgp9dx.streamlit.app)  


---

## 🧠 Features

✅ Predict whether a message is spam or not  
✅ Clean NLP preprocessing pipeline  
✅ Fast and lightweight  
✅ Trained on real-world SMS dataset  
✅ Easy-to-use Streamlit interface  

---

## 📂 Project Files

```
📁 SMS_Spam_Classifier/
│
├── app.py                # Streamlit application
├── SMS_spam_classifier.ipynb  # Notebook for EDA and model training
├── model.pkl             # Trained Naive Bayes model
├── vectorizer.pkl        # Trained TF-IDF vectorizer
├── spam.csv              # Original SMS dataset
├── requirements.txt      # Dependencies
└── .devcontainer/        # VS Code dev container config (optional)
```

---

## 🛠 Installation

### 🔗 Clone the Repository

```bash
git clone https://github.com/your-username/sms_spam_classifier.git
cd sms_spam_classifier
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶ Run the App

```bash
streamlit run app.py
```

---

## 📊 Dataset

📁 `spam.csv`  
- Source: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- 5,572 SMS messages labeled as **spam** or **ham** (not spam)

---

## 💡 How It Works

1. ✂️ **Preprocessing**: Lowercasing, stopword removal, stemming  
2. 🧠 **Vectorization**: TF-IDF transforms text into numerical features  
3. 🤖 **Model**: Multinomial Naive Bayes trained on processed data  
4. 🌐 **Interface**: Built using Streamlit for real-time predictions  

---

## 🔍 Examples

**📥 Input (Ham):**  
```
Hey, just checking in. Can we meet tomorrow?
```
**✅ Output:** Not Spam

---

**📥 Input (Spam):**  
```
You've won a free cruise! Click here to claim your prize now.
```
**🚫 Output:** Spam

---

## 🛠 Tech Stack

- 🐍 Python
- 📊 scikit-learn
- 🔤 NLTK
- 🧠 TF-IDF Vectorizer
- 🌐 Streamlit

---

## 👤 Author

- **Shreyak Mukherjee**  
  📧 [shreyakmukhrjeedgp@gmail.com](mailto:shreyakmukhrjeedgp@gmail.com)  
  🔗 [LinkedIn](https://www.linkedin.com/in/shreyak-mukherjee-203558275/)  
  💻 [GitHub](https://github.com/shreyakmukherjee)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

⭐ If you like this project, don't forget to give it a star on GitHub!
