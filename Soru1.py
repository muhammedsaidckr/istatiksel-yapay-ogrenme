import requests
import re
import nltk
from nltk.corpus import stopwords
import stanza
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline

# Web sitesinden haber metnini çekme fonksiyonu
def get_news_from_web(url):
    # Web sayfasına istek gönderme
    response = requests.get(url)

    # Sayfa başarılı şekilde alındıysa
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        news_text = " ".join([para.get_text() for para in paragraphs])
        return news_text
    else:
        return None


# Web sitesinin URL'sini girin
news_url = "https://www.hurriyet.com.tr/gundem/live-boludaki-otel-yangininda-can-kaybi-79-12nci-kattan-atlama-kararini-babasiyla-konusarak-almisti-sevvalden-kahreden-haber-42668147"
# Web'den haber metnini al
news_text = get_news_from_web(news_url)

# 2 . Standardization
text = news_text.lower()

# 3. Clean punctuations
text = re.sub(r'[^\w\s]|\d', '', text)

nltk.download('stopwords')
stanza.download('tr')

# 4.stopwords
stopwords_list = set(stopwords.words('turkish'))

# 5. Tokenization
words = text.split()
clean_words = [i for i in words if i not in stopwords_list]
clean_text = ' '.join(clean_words)

# 6
nlp = stanza.Pipeline('tr')
doc = nlp(clean_text)

for i in doc.sentences:
    for word in i.words:
        print(f'Word: {word.text}, Kok: {word.lemma}')

from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt

latest_text = ' '.join([word.lemma if word.lemma is not None else word.text for i in doc.sentences for word in i.words])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(latest_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

bounti = pipeline('sentiment-analysis', model="akoksal/bounti")

def analyze_sentiment(text):
    result = bounti(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

label, score = analyze_sentiment(clean_text[:512])

print(f'Label: {label}, Score: {score}')

# Label: neutral, Score: 0.9522287845611572
