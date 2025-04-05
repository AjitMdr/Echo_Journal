# mood_prediction_np/utils.py
import re
import pandas as pd

def clean_nepali_text(text):
    """Clean Nepali text data by removing special characters and extra whitespace."""
    if pd.isna(text):
        return ""
    
    # Remove special characters but keep Nepali Unicode characters
    text = re.sub(r'[^\u0900-\u097F\s]', ' ', str(text))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
