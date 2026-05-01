# 🔍 Fake News Detector — Complete Beginner's Guide

## 📁 Your Project Files

```
fake_news_project/
├── train_model.py       ← Run this FIRST (trains the AI)
├── app.py               ← Run this SECOND (starts the website)
└── templates/
    └── index.html       ← The website (auto-loaded by Flask)
```

---

## 🛠️ STEP 1 — Install Python

1. Go to: https://python.org/downloads
2. Click the big download button (Python 3.11 or newer)
3. Run the installer
4. ⚠️ IMPORTANT: Check the box "Add Python to PATH" before clicking Install
5. Click "Install Now"

---

## 🛠️ STEP 2 — Install VS Code

1. Go to: https://code.visualstudio.com
2. Download for your OS (Windows/Mac)
3. Install it (keep clicking Next / Continue)

---

## 🛠️ STEP 3 — Open Your Project Folder in VS Code

1. Open VS Code
2. Click File → Open Folder
3. Select your `fake_news_project` folder
4. Click "Open"

---

## 🛠️ STEP 4 — Open Terminal in VS Code

Press: Ctrl + ` (that's the backtick key, left of the 1 key)
OR go to: Terminal → New Terminal

---

## 🛠️ STEP 5 — Install Required Libraries

In the terminal, paste this and press Enter:

```
pip install pandas scikit-learn nltk flask
```

Wait for it to finish (takes 1-2 minutes). You'll see lots of text — that's normal.

---

## 🛠️ STEP 6 — Train the AI Model (Run ONCE)

In the terminal, type this and press Enter:

```
python train_model.py
```

You should see:
- "Model saved as model.pkl"
- An accuracy percentage

This creates a file called `model.pkl` — this is your trained AI!

---

## 🛠️ STEP 7 — Start the Website

In the terminal, type this and press Enter:

```
python app.py
```

You should see:
```
Fake News Detector is running!
Open your browser and go to:
  http://127.0.0.1:5000
```

---

## 🛠️ STEP 8 — Use the App!

1. Open your browser (Chrome, Firefox, Edge — any)
2. Go to: http://127.0.0.1:5000
3. Paste any news headline or article
4. Click "Analyze Article"
5. See the result!

---

## ❓ Common Problems & Solutions

**Problem: 'pip' is not recognized**
Solution: Restart your computer after installing Python, then try again.

**Problem: ModuleNotFoundError**
Solution: Run `pip install scikit-learn nltk flask` again in terminal.

**Problem: model.pkl not found**
Solution: Run `python train_model.py` first before `python app.py`.

**Problem: Port already in use**
Solution: Close any other terminal that might be running the app, or restart VS Code.

---

## 📊 What the Project Does (For Your Report)

| NLP Technique | Where Used |
|--------------|-----------|
| Text Filtration | clean_text() function removes noise, special chars, stopwords |
| Tokenization | text.split() breaks text into individual word tokens |
| N-gram Features | TfidfVectorizer(ngram_range=(1,2)) extracts unigrams & bigrams |
| Corpus Analysis | Training data of 40 labeled real/fake news samples |
| Text Classification | LogisticRegression predicts FAKE or REAL with probability |

---

## 🛑 To Stop the App

Press Ctrl + C in the terminal where app.py is running.
