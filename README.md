# Sentiment Analysis Web App

A comprehensive web application for sentiment analysis built with Flask, featuring advanced NLP techniques including emotion detection, aspect-based analysis, topic modeling, and web scraping capabilities.

## Features

- **Sentiment Analysis**: Analyze text sentiment using BERT-based models for accurate polarity scoring
- **Emotion Detection**: Identify emotions in text using pre-trained emotion classification models
- **Aspect-Based Analysis**: Perform sentiment analysis on specific aspects/nouns in the text
- **Topic Modeling**: Extract key topics from text using Latent Dirichlet Allocation (LDA)
- **Word Highlighting**: Visualize positive and negative words in the analyzed text
- **Web Scraping**: Extract and analyze content from web pages
- **Data Visualization**: Interactive dashboard with charts for sentiment distribution, polarity trends, emotion analysis, and more
- **Data Export**: Export analysis history and scraped data to Excel files
- **REST API**: Programmatic access to sentiment analysis functionality
- **Responsive Design**: Mobile-friendly web interface

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sentiment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data (required for text processing):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('averaged_perceptron_tagger')
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

### Web Interface
1. **Text Analysis**: Enter text in the main form and click "Analyze Sentiment" to get comprehensive analysis results
2. **Web Scraping**: Provide a URL to scrape and analyze web content automatically
3. **Visualization**: Click "View Sentiment Visualization" to access the interactive dashboard
4. **Export**: Use "Export History" to download your analysis history as an Excel file

### API Usage
The application provides a REST API endpoint for programmatic access:

**Endpoint**: `POST /api/analyze`

**Request Body**:
```json
{
  "text": "Your text to analyze here"
}
```

**Response**:
```json
{
  "sentiment": "POSITIVE",
  "polarity": 0.9876
}
```

## API Endpoints

- `GET /` - Main application interface
- `POST /analyze` - Analyze text sentiment (web interface)
- `POST /api/analyze` - Analyze text sentiment (API)
- `GET /visualize` - Sentiment visualization dashboard
- `GET /export` - Export analysis history to Excel
- `GET /export_scraped` - Export scraped data to Excel
- `POST /scrape` - Scrape and analyze web content

## Technologies Used

- **Backend**: Flask (Python web framework)
- **NLP Models**:
  - BERT (via transformers library) for sentiment analysis
  - Emotion classification model (j-hartmann/emotion-english-distilroberta-base)
- **Text Processing**: NLTK, TextBlob
- **Machine Learning**: scikit-learn (for topic modeling)
- **Web Scraping**: BeautifulSoup4, requests
- **Data Processing**: pandas
- **Visualization**: Chart.js
- **Frontend**: HTML5, CSS3, JavaScript

## Project Structure

```
sentiment/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── test_app.py           # Unit tests
├── static/
│   └── logo.jpg          # Application logo
└── templates/
    ├── index.html        # Main interface
    └── visualize.html    # Visualization dashboard
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library for NLP models
- NLTK for natural language processing
- Chart.js for data visualization
- Flask framework for web development
