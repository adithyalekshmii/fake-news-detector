import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0a0a0f; }

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #12121a 100%);
    color: #e8e8f0;
}

.title-text {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #6c63ff, #ff6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: #7070a0;
    font-size: 16px;
    margin-bottom: 30px;
}

.result-real {
    background: rgba(0,229,160,0.08);
    border: 1.5px solid rgba(0,229,160,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.result-fake {
    background: rgba(255,77,109,0.08);
    border: 1.5px solid rgba(255,77,109,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.verdict-real {
    font-family: 'Syne', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #00e5a0;
}

.verdict-fake {
    font-family: 'Syne', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #ff4d6d;
}
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    
    training_data = [
        ("Scientists confirmed vaccine effectiveness after extensive clinical trials involving 40000 participants.", "REAL"),
        ("The stock market closed higher as investors reacted positively to the latest employment data.", "REAL"),
        ("The government announced a new infrastructure plan worth 1.2 trillion to repair roads and bridges.", "REAL"),
        ("Researchers at MIT developed a new algorithm that improves battery efficiency by 30 percent.", "REAL"),
        ("The central bank raised interest rates by 0.25 percent to combat rising inflation.", "REAL"),
        ("Local authorities confirmed the wildfire has been contained after three days of firefighting.", "REAL"),
        ("The United Nations released a report warning about accelerating climate change impacts.", "REAL"),
        ("A new study published in Nature found a link between sleep deprivation and health risks.", "REAL"),
        ("The prime minister addressed parliament regarding the new education reform bill.", "REAL"),
        ("NASA successfully launched its new telescope to observe distant galaxies.", "REAL"),
        ("The hospital reported a decline in patient admissions following the vaccination drive.", "REAL"),
        ("Olympic athletes broke three world records during the championship event held in Paris.", "REAL"),
        ("The company quarterly earnings report showed a 15 percent increase in revenue.", "REAL"),
        ("Archaeologists discovered ancient ruins estimated to be over 3000 years old in Egypt.", "REAL"),
        ("The Supreme Court ruled on the landmark case regarding privacy rights.", "REAL"),
        ("A new law was passed requiring companies to disclose carbon emissions annually.", "REAL"),
        ("The city council approved a budget increase for public transportation infrastructure.", "REAL"),
        ("Medical experts recommend annual checkups to detect early signs of chronic diseases.", "REAL"),
        ("The trade agreement between the two nations was signed after months of negotiations.", "REAL"),
        ("University researchers published findings on the impact of social media on mental health.", "REAL"),
        ("SHOCKING Government secretly putting mind control chips in COVID vaccines whistleblower reveals!", "FAKE"),
        ("Celebrities are all reptilian aliens in disguise leaked documents prove beyond doubt!", "FAKE"),
        ("Drinking bleach cures cancer instantly Big Pharma does not want you to know this secret!", "FAKE"),
        ("The moon landing was faked in a Hollywood studio NASA finally admits the truth!", "FAKE"),
        ("5G towers spread viruses scientist who exposed the truth found dead in suspicious circumstances!", "FAKE"),
        ("Eating chocolate every day makes you lose 50 pounds in a week doctors are furious!", "FAKE"),
        ("World leaders are part of a secret society controlling all global events from underground bunkers!", "FAKE"),
        ("This miracle herb cures diabetes cancer and COVID in 24 hours Doctors hate this one trick!", "FAKE"),
        ("BREAKING Famous celebrity arrested for running an underground reptile smuggling ring!", "FAKE"),
        ("The government is hiding aliens at Area 51 an insider with proof has gone missing!", "FAKE"),
        ("Scientists BANNED from revealing the truth about flat earth by global elite!", "FAKE"),
        ("New study PROVES that WiFi signals cause brain tumors Share before they delete this!", "FAKE"),
        ("Politician secretly a robot confirmed by undercover footage mainstream media silent!", "FAKE"),
        ("You will not BELIEVE what they found in tap water The truth the government is hiding!", "FAKE"),
        ("Ancient pyramid discovered on Mars proves aliens built Egyptian pyramids NASA covers it up!", "FAKE"),
        ("This banned natural remedy destroys all viruses instantly Pharmaceutical companies furious!", "FAKE"),
        ("URGENT All phones will explode on midnight unless you forward this message to everyone!", "FAKE"),
        ("Secret underground city discovered beneath major capital government denies existence!", "FAKE"),
        ("Local man cures blindness by staring at the sun for 10 minutes Doctors baffled!", "FAKE"),
        ("Worlds richest people are funding a plan to reduce global population leaked emails show!", "FAKE"),
    ]
    
    texts = [clean_text(t[0]) for t in training_data]
    labels = [t[1] for t in training_data]
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(texts, labels)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

model = load_model()

st.markdown('<div class="title-text">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any news article or headline below to check if it\'s Real or Fake</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Method", "NLP")
with col2:
    st.metric("Features", "TF-IDF")
with col3:
    st.metric("Model", "ML")

st.markdown("---")

st.markdown("**🧪 Try an example:**")
col_r, col_f = st.columns(2)
with col_r:
    if st.button("✅ Real News Example"):
        st.session_state.example = "Scientists have confirmed the effectiveness of the new vaccine after extensive clinical trials involving 40,000 participants across multiple countries."
with col_f:
    if st.button("❌ Fake News Example"):
        st.session_state.example = "SHOCKING: Government secretly putting mind control chips in COVID vaccines, a whistleblower reveals! Big Pharma doesn't want you to know this secret!"

default_text = st.session_state.get('example', '')
news_input = st.text_area(
    "📰 Paste News Article or Headline",
    value=default_text,
    height=180,
    placeholder="Paste a news headline or full article here to analyze it..."
)

word_count = len(news_input.strip().split()) if news_input.strip() else 0
st.caption(f"{word_count} words {'✅' if word_count >= 5 else '⚠️ Need at least 5 words'}")

if st.button("🔍 Analyze Article", use_container_width=True, type="primary"):
    if not news_input.strip():
        st.error("Please paste a news article first.")
    elif word_count < 5:
        st.error("Please enter at least 5 words for a proper analysis.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(news_input)
            prediction = model.predict([cleaned])[0]
            probs = model.predict_proba([cleaned])[0]
            classes = list(model.classes_)
            real_prob = round(probs[classes.index('REAL')] * 100, 1)
            fake_prob = round(probs[classes.index('FAKE')] * 100, 1)

        st.markdown("### Result")

        if prediction == "REAL":
            st.markdown(f'''
            <div class="result-real">
                <div style="font-size:48px">✅</div>
                <div class="verdict-real">Real News</div>
                <div style="color:#7070a0;margin-top:8px">Confidence: {real_prob}%</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-fake">
                <div style="font-size:48px">❌</div>
                <div class="verdict-fake">Fake News</div>
                <div style="color:#7070a0;margin-top:8px">Confidence: {fake_prob}%</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probability Breakdown:**")
        st.progress(real_prob / 100, text=f"✅ Real News: {real_prob}%")
        st.progress(fake_prob / 100, text=f"❌ Fake News: {fake_prob}%")

st.markdown("---")
st.caption("Fake News Detection · Made by Adithyalekshmi")
