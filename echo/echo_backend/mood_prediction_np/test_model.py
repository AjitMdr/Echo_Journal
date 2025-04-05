"""
Test script for evaluating the Nepali BERT sentiment analysis model on custom text.

Usage:
    python test_model.py "Your Nepali text here"
"""

from mood_prediction_np.utils import clean_nepali_text
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel
import re
from django.conf import settings
import django

# Initialize Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
django.setup()

# Import after Django setup

# Model parameters
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 128

# Define model and tokenizer paths - adjust based on your folder structure
MODEL_DIR = os.path.join(settings.BASE_DIR, 'bert_model', 'model', 'model')

# Define the SentimentClassifier class


class SentimentClassifier(torch.nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


def get_latest_model_path():
    """Get the path of the latest trained model."""
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory not found: {MODEL_DIR}")
        return None

    model_dirs = [d for d in os.listdir(MODEL_DIR) if d.startswith('model_')]
    if not model_dirs:
        print(f"No model directories found in {MODEL_DIR}")
        return None

    latest_model_dir = sorted(model_dirs)[-1]  # Get the most recent model
    print(f"Using model directory: {latest_model_dir}")

    # Look for model.pt file
    model_pt_path = os.path.join(MODEL_DIR, latest_model_dir, 'model.pt')
    if os.path.exists(model_pt_path):
        return {
            'model_path': model_pt_path,
            'tokenizer_path': os.path.join(MODEL_DIR, latest_model_dir)
        }

    # If model.pt not found, check if at least tokenizer files exist
    tokenizer_files = ['vocab.txt', 'tokenizer_config.json']
    if all(os.path.exists(os.path.join(MODEL_DIR, latest_model_dir, file)) for file in tokenizer_files):
        print(f"Model weights file not found, but tokenizer files exist.")
        return {
            'model_path': None,
            'tokenizer_path': os.path.join(MODEL_DIR, latest_model_dir)
        }

    print(f"No model files found in {latest_model_dir}")
    return None


def predict_sentiment_rule_based(text, tokenizer):
    """Enhanced rule-based sentiment analysis when model weights are not available."""
    cleaned_text = clean_nepali_text(text)

    # Enhanced rule-based sentiment analysis with more comprehensive word lists
    # Negative words
    negative_words = [
        "दुःख", "पीडा", "रिस", "क्रोध", "भय", "चिन्ता", "नराम्रो", "खराब",
        "निराश", "कष्ट", "द्वेष", "घृणा", "दुर्भाग्य", "पीडित", "असफल", "बिरामी",
        "गरिब", "दुर्बल", "मृत्यु", "अस्वीकार", "हार", "टुटेको", "दुःखी", "रुन्",
        "कडा", "गाली", "आलोचना", "नकारात्मक", "विपत्ति", "खतरा", "समस्या"
    ]

    # Positive words - adding more positive words and word stems
    positive_words = [
        "खुशी", "आनन्द", "माया", "प्रेम", "सुख", "शान्ति", "राम्रो", "मिठो",
        "उत्तम", "सफलता", "उत्साह", "हाँसो", "आशा", "विश्वास", "आदर", "सहयोग",
        "सकारात्मक", "उपलब्धि", "सफल", "जित", "प्रशंसा", "सुन्दर", "स्वस्थ",
        "शक्तिशाली", "समृद्ध", "खुसी", "असल", "मन पर्यो", "धन्यवाद", "आभार",
        "उत्कृष्ट", "धेरै राम्रो", "सुखद", "ताजगी", "प्रफुल्लित", "उत्साहजनक",
        "प्रगति", "प्रोत्साहित", "उज्यालो", "नयाँ", "सक्छु", "सफलतासाथ", "मिलेर"
    ]

    # Common neutral verbs that shouldn't affect sentiment
    neutral_words = ["भयो", "थियो", "हो", "गरें",
                     "गर्दा", "छ", "थिए", "हुन्छ", "भए"]

    # Debug print to check the text
    print(f"Rule-based analysis for: '{cleaned_text}'")

    # Tokenize the text
    original_tokens = tokenizer.tokenize(cleaned_text.lower())

    # Filter out neutral words and get stemmed tokens for better matching
    filtered_tokens = []
    for token in original_tokens:
        # Skip neutral words
        if any(token.startswith(neutral) for neutral in neutral_words):
            continue
        filtered_tokens.append(token)

    # Direct word matching for exact matches
    # Check for positive and negative words and print matches for debugging
    neg_matches = []
    pos_matches = []

    # First, check full text for phrases and word forms that may be split by tokenizer
    for pos_word in positive_words:
        if pos_word in cleaned_text:
            pos_matches.append(pos_word)

    for neg_word in negative_words:
        if neg_word in cleaned_text:
            neg_matches.append(neg_word)

    # Then check individual tokens
    for token in filtered_tokens:
        # Check if token is a negative word or contains a negative word stem
        if any(token.startswith(neg) or neg in token for neg in negative_words):
            neg_matches.append(token)
        # Check if token is a positive word or contains a positive word stem
        if any(token.startswith(pos) or pos in token for pos in positive_words):
            pos_matches.append(token)

    # Remove duplicates
    neg_matches = list(set(neg_matches))
    pos_matches = list(set(pos_matches))

    print(f"Original tokens: {original_tokens}")
    print(f"Filtered tokens: {filtered_tokens}")
    print(f"Negative matches: {neg_matches}")
    print(f"Positive matches: {pos_matches}")

    neg_count = len(neg_matches)
    pos_count = len(pos_matches)

    # Check for negation words that might invert the sentiment
    negation_words = ["छैन", "होइन", "नहुने", "बिना", "न"]
    has_negation = any(neg in cleaned_text for neg in negation_words)

    # If negation is present, consider inverting the sentiment
    if has_negation:
        print("Negation detected, adjusting sentiment")
        # Check if negation is applied to positive or negative words
        # This is a simplified approach; a more sophisticated approach would check proximity
        if pos_count > neg_count:
            # If more positive words and negation present, might be negative
            neg_count += 1
        elif neg_count > pos_count:
            # If more negative words and negation present, might be positive
            pos_count += 1

    if neg_count > pos_count:
        sentiment = "Negative"
        confidence = 0.5 + min((neg_count - pos_count) * 0.1, 0.4)
    elif pos_count > neg_count:
        sentiment = "Positive"
        confidence = 0.5 + min((pos_count - neg_count) * 0.1, 0.4)
    else:
        sentiment = "Neutral"
        confidence = 0.5

    print(
        f"Final rule-based sentiment: {sentiment} with confidence {confidence:.2f}")

    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'confidence': confidence,
        'rule_based': True,
        'pos_matches': pos_matches,
        'neg_matches': neg_matches
    }


def predict_with_model(text, model_info):
    """Predict sentiment using the pre-trained BERT model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Clean the text
    cleaned_text = clean_nepali_text(text)
    print(f"Cleaned text: {cleaned_text}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_info['tokenizer_path'])

    # If no model weights are available, use rule-based approach
    if model_info['model_path'] is None:
        print("Model weights not found. Using rule-based approach.")
        return predict_sentiment_rule_based(text, tokenizer)

    # Load model
    model = SentimentClassifier(MODEL_NAME)
    model.load_state_dict(torch.load(
        model_info['model_path'], map_location=device))
    model.to(device)
    model.eval()

    # Tokenize input
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # Get probabilities for all classes
        all_probs = probabilities[0].cpu().numpy()

    # Map prediction to sentiment (0: Negative, 1: Neutral, 2: Positive)
    sentiment_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    result = {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment_map[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'Negative': float(all_probs[0]),
            'Neutral': float(all_probs[1]),
            'Positive': float(all_probs[2])
        },
        'rule_based': False
    }

    return result


def main():
    # Check if a text argument is provided
    if len(sys.argv) < 2:
        print("Usage: python test_model.py \"Your Nepali text here\"")
        print("\nAlternatively, enter a text when prompted.")
        text = input("\nEnter Nepali text to analyze: ")
    else:
        text = " ".join(sys.argv[1:])

    if not text.strip():
        print("Error: Please provide a text to analyze.")
        return

    # Get the latest model path
    model_info = get_latest_model_path()
    if not model_info:
        print("No model found. Please train a model first or check the model directory.")
        return

    # Predict sentiment
    result = predict_with_model(text, model_info)

    # Display result
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS RESULT")
    print("="*50)
    print(f"Input text: {result['text']}")
    print(f"Cleaned text: {result['cleaned_text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(
        f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")

    if 'probabilities' in result:
        print("\nClass probabilities:")
        for sentiment, prob in result['probabilities'].items():
            print(f"  {sentiment}: {prob:.4f} ({prob*100:.1f}%)")

    if result['rule_based']:
        print("\nNote: Using rule-based analysis (model weights not available)")
        if 'pos_matches' in result:
            print(
                f"Positive matches: {', '.join(result['pos_matches']) if result['pos_matches'] else 'None'}")
        if 'neg_matches' in result:
            print(
                f"Negative matches: {', '.join(result['neg_matches']) if result['neg_matches'] else 'None'}")

    print("="*50)


if __name__ == "__main__":
    main()
