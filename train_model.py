# ============================================================
# FAKE NEWS DETECTOR - MODEL TRAINING SCRIPT
# Run this ONCE to create your AI model
# How to run: Open terminal, type: python train_model.py
# ============================================================

import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data (runs only first time)
print("Downloading language data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

import re
from nltk.corpus import stopwords

# -----------------------------------------------
# SAMPLE TRAINING DATA
# In a real project you'd load a CSV dataset.
# This gives you a working model right away.
# -----------------------------------------------
training_data = [
    # REAL news samples
    ("Scientists have confirmed the effectiveness of the new vaccine after extensive clinical trials involving 40,000 participants.", "REAL"),
    ("The stock market closed higher on Friday as investors reacted positively to the latest employment data.", "REAL"),
    ("The government announced a new infrastructure plan worth $1.2 trillion to repair roads and bridges.", "REAL"),
    ("Researchers at MIT developed a new algorithm that improves battery efficiency by 30 percent.", "REAL"),
    ("The central bank raised interest rates by 0.25 percent to combat rising inflation.", "REAL"),
    ("Local authorities confirmed that the wildfire has been contained after three days of firefighting efforts.", "REAL"),
    ("The United Nations released a report warning about accelerating climate change impacts.", "REAL"),
    ("A new study published in Nature found a link between sleep deprivation and increased health risks.", "REAL"),
    ("The prime minister addressed parliament regarding the new education reform bill.", "REAL"),
    ("NASA successfully launched its new telescope to observe distant galaxies.", "REAL"),
    ("The hospital reported a significant decline in patient admissions following the vaccination drive.", "REAL"),
    ("Olympic athletes broke three world records during the championship event held in Paris.", "REAL"),
    ("The company's quarterly earnings report showed a 15 percent increase in revenue.", "REAL"),
    ("Archaeologists discovered ancient ruins estimated to be over 3,000 years old in Egypt.", "REAL"),
    ("The Supreme Court ruled on the landmark case regarding privacy rights.", "REAL"),
    ("A new law was passed requiring companies to disclose carbon emissions annually.", "REAL"),
    ("The city council approved a budget increase for public transportation infrastructure.", "REAL"),
    ("Medical experts recommend annual checkups to detect early signs of chronic diseases.", "REAL"),
    ("The trade agreement between the two nations was signed after months of negotiations.", "REAL"),
    ("University researchers published findings on the impact of social media on mental health.", "REAL"),

    # FAKE news samples
    ("SHOCKING: Government secretly putting mind control chips in COVID vaccines, whistleblower reveals!", "FAKE"),
    ("Celebrities are all reptilian aliens in disguise, leaked documents prove beyond doubt!!!", "FAKE"),
    ("Drinking bleach cures cancer instantly! Big Pharma doesn't want you to know this secret!", "FAKE"),
    ("The moon landing was faked in a Hollywood studio, NASA finally admits the truth!", "FAKE"),
    ("5G towers spread viruses, scientist who exposed the truth found dead in suspicious circumstances!", "FAKE"),
    ("Eating chocolate every day makes you lose 50 pounds in a week, doctors are furious!", "FAKE"),
    ("World leaders are part of a secret society controlling all global events from underground bunkers!", "FAKE"),
    ("This miracle herb cures diabetes, cancer, and COVID in 24 hours! Doctors hate this one trick!", "FAKE"),
    ("BREAKING: Famous celebrity arrested for running an underground reptile smuggling ring!!!", "FAKE"),
    ("The government is hiding aliens at Area 51, an insider with proof has gone missing!", "FAKE"),
    ("Scientists BANNED from revealing the truth about flat earth by global elite!", "FAKE"),
    ("New study PROVES that WiFi signals cause brain tumors! Share before they delete this!", "FAKE"),
    ("Politician secretly a robot confirmed by undercover footage, mainstream media silent!", "FAKE"),
    ("You won't BELIEVE what they found in tap water! The truth the government is hiding!", "FAKE"),
    ("Ancient pyramid discovered on Mars proves aliens built Egyptian pyramids, NASA covers it up!", "FAKE"),
    ("This banned natural remedy destroys all viruses instantly! Pharmaceutical companies furious!", "FAKE"),
    ("URGENT: All phones will explode on midnight unless you forward this message to everyone!", "FAKE"),
    ("Secret underground city discovered beneath major capital, government denies existence!", "FAKE"),
    ("Local man cures blindness by staring at the sun for 10 minutes! Doctors baffled!", "FAKE"),
    ("World's richest people are funding a plan to reduce global population, leaked emails show!", "FAKE"),
]

print(f"Total training samples: {len(training_data)}")

# Separate texts and labels
texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

# -----------------------------------------------
# TEXT CLEANING FUNCTION
# This is the "Text Filtration" step from your project
# -----------------------------------------------
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common English stopwords (like "the", "is", "a")
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Clean all texts
print("Cleaning text data...")
cleaned_texts = [clean_text(t) for t in texts]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_texts, labels, test_size=0.2, random_state=42
)

# -----------------------------------------------
# BUILD THE MODEL PIPELINE
# TF-IDF = converts text to numbers (N-gram features)
# Logistic Regression = classifies as FAKE or REAL
# -----------------------------------------------
print("Training the model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),   # Uses 1-gram and 2-gram features
        max_features=5000,     # Consider top 5000 word patterns
        min_df=1
    )),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

# -----------------------------------------------
# EVALUATE THE MODEL
# -----------------------------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print("="*50)
print(f"Accuracy: {accuracy*100:.1f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# -----------------------------------------------
# SAVE THE MODEL
# This creates a file 'model.pkl' that the web app uses
# -----------------------------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'model.pkl'")
print("\nNow run: python app.py")
print("Then open: http://127.0.0.1:5000 in your browser")
