import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Specify custom NLTK path if needed
nltk.data.path.append('/Users/your_username/nltk_data')
nltk.download('punkt')
nltk.download('stopwords')

# Load and clean data
data = pd.read_csv("../feedback.csv")
data['VoteChoice'] = data['VoteChoice'].fillna('').astype(str)

# Define themes with keywords
themes = {
    "Experience": ["experience", "feel", "impression", "overall"],
    "Product Style": ["style", "design", "trendy", "quality", "look", "fit", "material"],
    "Customer Service": ["service", "support", "staff", "helpful", "friendly", "attitude"]
}

# Define sentiment keywords for positive and negative classification
positive_keywords = ["helpful", "friendly", "excellent", "satisfied", "good", "polite"]
negative_keywords = ["rude", "unhelpful", "poor", "bad", "dissatisfied", "unhappy"]

# Initialize sentence-transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)


# Apply preprocessing
data['processed_text'] = data['VoteChoice'].apply(preprocess_text)

# Generate embeddings for keywords in each theme and sentiment category
theme_embeddings = {theme: model.encode(keywords) for theme, keywords in themes.items()}
positive_embeddings = model.encode(positive_keywords)
negative_embeddings = model.encode(negative_keywords)

# Initialize sentiment analyzer for general polarity scoring
sentiment_analyzer = SentimentIntensityAnalyzer()


# Function to classify theme based on semantic similarity
def classify_theme(text):
    text_embedding = model.encode(text)
    max_similarity = 0
    assigned_theme = "Other"

    # Compare text with each theme's keywords using cosine similarity
    for theme, embeddings in theme_embeddings.items():
        similarity_scores = util.cos_sim(text_embedding, embeddings)
        avg_similarity = similarity_scores.mean().item()  # Average similarity for theme
        if avg_similarity > max_similarity:
            max_similarity = avg_similarity
            assigned_theme = theme

    return assigned_theme


# Function to determine sentiment with a hybrid approach
def classify_sentiment(text):
    text_embedding = model.encode(text)
    # Cosine similarity checks for positive and negative sentiments
    pos_similarity = util.cos_sim(text_embedding, positive_embeddings).mean().item()
    neg_similarity = util.cos_sim(text_embedding, negative_embeddings).mean().item()
    sentiment_score = sentiment_analyzer.polarity_scores(text)['compound']

    # Logic to determine final sentiment
    if pos_similarity > neg_similarity and pos_similarity > 0.6:
        return "Positive"
    elif neg_similarity > pos_similarity and neg_similarity > 0.6:
        return "Negative"
    elif sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Apply theme and sentiment classification
data['Theme'] = data['processed_text'].apply(classify_theme)
data['Sentiment'] = data['VoteChoice'].apply(classify_sentiment)

# Regional analysis by theme and sentiment
region_theme_summary = data.groupby(['Region', 'Location', 'Theme']).size().unstack(fill_value=0)
region_sentiment_summary = data.groupby(['Region', 'Location', 'Sentiment']).size().unstack(fill_value=0)

# Save results to CSV
region_theme_summary.to_csv("themes_by_region_location.csv")
region_sentiment_summary.to_csv("sentiment_by_region_location.csv")

print("Analysis complete. Data saved to themes_by_region_location.csv and sentiment_by_region_location.csv")
