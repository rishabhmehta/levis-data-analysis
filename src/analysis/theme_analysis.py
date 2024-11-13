import pandas as pd
from src.utils.data_loader import load_feedback_data


# Define hardcoded themes and sentiment keywords
themes = {
    "Satisfaction": ["satisfied", "dissatisfied", "happy", "unhappy", "good", "bad", "excellent", "poor"],
    "Product Style": ["style", "design", "look", "fashion", "trendy", "color", "material", "quality"],
    "Customer Service": ["service", "staff", "helpful", "support", "assist", "attitude", "friendly", "rude", "helping", "thank you", "appreciate", "polite"]
}

positive_keywords = ["thank you", "helpful", "appreciate", "polite", "friendly", "excellent", "satisfied", "happy"]
negative_keywords = ["rude", "unhelpful", "poor", "bad", "dissatisfied", "angry", "unhappy"]

# Function to classify feedback based on themes
def classify_feedback(feedback, themes):
    feedback_themes = []
    for theme, keywords in themes.items():
        if any(keyword in feedback.lower() for keyword in keywords):
            feedback_themes.append(theme)
    return feedback_themes if feedback_themes else ["Other"]

# Function to detect sentiment in feedback
def detect_sentiment(feedback):
    feedback_lower = feedback.lower()
    if any(pos_kw in feedback_lower for pos_kw in positive_keywords):
        return "Positive"
    elif any(neg_kw in feedback_lower for neg_kw in negative_keywords):
        return "Negative"
    else:
        return "Neutral"

# Load data
data = load_feedback_data()

# Ensure VoteChoice values are strings and handle missing values
data['VoteChoice'] = data['VoteChoice'].fillna('').astype(str)

# Apply theme classification and sentiment detection
data['Themes'] = data['VoteChoice'].apply(lambda x: classify_feedback(x, themes))
data['Sentiment'] = data['VoteChoice'].apply(detect_sentiment)

# Select relevant columns and save to CSV
output_data = data[['VoteChoice', 'Themes', 'Sentiment', 'Location', 'Region']]
output_data.to_csv("theme_sentiment_analysis_with_location.csv", index=False)

print("Theme and sentiment analysis complete. Results saved to theme_sentiment_analysis_with_location.csv")
