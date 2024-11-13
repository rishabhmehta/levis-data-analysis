import pandas as pd

def load_feedback_data(feedback_path="../feedback.csv", keywords_path="../feedback_with_keywords.csv"):
    feedback_data = pd.read_csv(feedback_path)
    keywords_data = pd.read_csv(keywords_path)
    merged_data = pd.merge(feedback_data, keywords_data, on="VoteChoice", how="inner")
    return merged_data
