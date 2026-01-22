from flask import Flask, render_template, request, session, jsonify, send_file
from textblob import TextBlob
from transformers import pipeline
import requests
import re
import json
import nltk
from nltk import word_tokenize, pos_tag
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # For session management

# Load NLP models
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
sentiment_classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '').strip()
    if not text:
        return render_template('index.html', error="Please enter some text to analyze.")
    bert_result = sentiment_classifier(text[:512])
    sentiment = bert_result[0]['label']
    polarity = bert_result[0]['score']

    blob = TextBlob(text)

    # Word highlighting
    highlighted_text = highlight_sentiment_words(text, blob)

    # Aspect-based analysis
    aspects = aspect_sentiment_analysis(text)

    # Emotion detection
    emotions = emotion_classifier(text[:512])  # Limit text length for model

    # Topic modeling
    topics = extract_topics(text)

    # Store in session for visualization
    if 'history' not in session:
        session['history'] = []
    session['history'].append({
        'text': text,
        'sentiment': sentiment,
        'polarity': polarity,
        'emotions': emotions,
        'aspects': aspects,
        'topics': topics
    })

    return render_template('index.html', sentiment=sentiment, polarity=polarity, text=text,
                           highlighted_text=highlighted_text, aspects=aspects, emotions=emotions,
                           topics=topics)

def highlight_sentiment_words(text, blob):
    words = blob.words
    highlighted = []
    for word in words:
        polarity = TextBlob(word).sentiment.polarity
        if polarity > 0:
            highlighted.append(f'<span class="positive">{word}</span>')
        elif polarity < 0:
            highlighted.append(f'<span class="negative">{word}</span>')
        else:
            highlighted.append(word)
    return ' '.join(highlighted)

def aspect_sentiment_analysis(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    aspects = {}
    for word, tag in tagged:
        if tag.startswith('NN'):  # Nouns
            sentiment = TextBlob(word).sentiment.polarity
            aspects[word] = 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
    return aspects

def extract_topics(text, num_topics=1):
    if not text.strip():
        return []
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-5 - 1:-1]]
        topics.append(' '.join(top_words))
    return topics

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400
    bert_result = sentiment_classifier(text[:512])
    sentiment = bert_result[0]['label']
    polarity = bert_result[0]['score']
    return jsonify({'sentiment': sentiment, 'polarity': polarity})

@app.route('/export')
def export():
    history = session.get('history', [])
    if not history:
        return "No data to export", 400
    # Simplify data for export
    export_data = []
    for h in history:
        export_data.append({
            'text': h['text'],
            'sentiment': h['sentiment'],
            'polarity': h['polarity'],
            'emotions': str(h['emotions']),
            'aspects': str(h['aspects']),
            'topics': str(h['topics'])
        })
    df = pd.DataFrame(export_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sentiment History')
    output.seek(0)
    return send_file(output, download_name='sentiment_history.xlsx', as_attachment=True)

@app.route('/export_scraped')
def export_scraped():
    scraped_data = session.get('scraped_data', [])
    if not scraped_data:
        return "No scraped data to export", 400
    df = pd.DataFrame(scraped_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Scraped Data')
    output.seek(0)
    return send_file(output, download_name='scraped_data.xlsx', as_attachment=True)

def map_sentiment(label):
    if label in ['5 star', '4 star']:
        return 'Positive'
    elif label == '3 star':
        return 'Neutral'
    else:
        return 'Negative'

@app.route('/visualize')
def visualize():
    history = session.get('history', [])
    if not history:
        data_json = 'null'
    else:
        sentiments = [map_sentiment(h['sentiment']) for h in history]
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        neutral_count = sentiments.count('Neutral')
        polarities = [h['polarity'] for h in history]
        emotions = [h['emotions'][0]['label'] for h in history if h['emotions'] and len(h['emotions']) > 0]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        aspects = {}
        for h in history:
            for aspect, sent in h['aspects'].items():
                if aspect not in aspects:
                    aspects[aspect] = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
                aspects[aspect][sent] += 1
        topics_list = []
        for h in history:
            topics_list.extend(h['topics'])
        data = {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'polarities': polarities,
            'emotion_counts': emotion_counts,
            'aspects': aspects,
            'topics': topics_list
        }
        data_json = json.dumps(data)
    return render_template('visualize.html', data_json=data_json)

@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form.get('url')
    if not url:
        return render_template('index.html', scrape_message="No URL provided.")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        text = response.text
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        paragraphs = soup.find_all('p')
        scraped_data = []
        for i, p in enumerate(paragraphs, 1):
            content = p.get_text(strip=True)
            # Clean and filter content
            content = clean_text(content)
            if content and len(content.split()) > 5:  # Filter short paragraphs
                scraped_data.append({
                    'index': i,
                    'content': content,
                    'word_count': len(content.split()),
                    'char_count': len(content)
                })
        if not scraped_data:
            return render_template('index.html', scrape_message="No suitable paragraphs found on the provided URL.")
        scraped_table = {'headers': ['Index', 'Content', 'Word Count', 'Char Count'], 'rows': [[d['index'], d['content'], d['word_count'], d['char_count']] for d in scraped_data]}
        # Store structured scraped data in session for export
        session['scraped_data'] = scraped_data
        return render_template('index.html', scraped_tables=[scraped_table])
    except requests.exceptions.RequestException as e:
        return render_template('index.html', scrape_message=f"Error accessing URL: {str(e)}")
    except Exception as e:
        return render_template('index.html', scrape_message=f"Error scraping: {str(e)}")

def clean_text(text):
    """Clean scraped text by removing extra whitespace and HTML artifacts."""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-breaking spaces and other HTML entities
    text = text.replace('\u00a0', ' ').replace('\n', ' ').replace('\r', ' ')
    return text

if __name__ == '__main__':
    app.run(debug=True)
