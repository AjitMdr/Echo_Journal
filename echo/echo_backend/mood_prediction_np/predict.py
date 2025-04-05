import torch
from transformers import AutoTokenizer, AutoModel
import os
from django.conf import settings
from mood_prediction_np.utils import clean_nepali_text
import re

# Define model and tokenizer paths
# Updated to match the nested directory structure
MODEL_DIR = os.path.join(settings.BASE_DIR, 'bert_model', 'model', 'model')


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
    print(f"Found latest model directory: {latest_model_dir}")

    # Look for model.pt file
    model_pt_path = os.path.join(MODEL_DIR, latest_model_dir, 'model.pt')
    if os.path.exists(model_pt_path):
        return model_pt_path

    # If model.pt is not found, check if tokenizer files exist which indicates a valid model directory
    tokenizer_files = ['vocab.txt', 'tokenizer_config.json']
    if all(os.path.exists(os.path.join(MODEL_DIR, latest_model_dir, file)) for file in tokenizer_files):
        # Model file not found but tokenizer exists - this is a directory with only tokenizer files
        # from Colab training that didn't save the model.pt file correctly
        print(
            f"Model weights file not found, but tokenizer files exist in {latest_model_dir}")
        # Return the directory so we can at least use the tokenizer
        return os.path.join(MODEL_DIR, latest_model_dir)

    print(f"No model files found in {latest_model_dir}")
    return None


# Get the model path
MODEL_PATH = get_latest_model_path()
print(f"Using model path: {MODEL_PATH}")


class SentimentClassifier(torch.nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased", num_classes=3):
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


def predict_sentiment(text):
    """
    Predict sentiment of Nepali text using BERT model.
    Returns one of: 'Negative', 'Neutral', 'Positive'
    """
    try:
        if not MODEL_PATH:
            print("No trained model found. Please train the model first.")
            return {"sentiment": "Neutral", "confidence": 0.33, "error": "No model found"}

        # Clean the text first
        cleaned_text = clean_nepali_text(text)
        if not cleaned_text:
            print("Text is empty after cleaning")
            return {"sentiment": "Neutral", "confidence": 0.33, "error": "Empty text after cleaning"}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Check if MODEL_PATH is a directory (contains only tokenizer) or a file (full model)
        if os.path.isdir(MODEL_PATH):
            # Only tokenizer is available, use rule-based fallback
            print("Using rule-based fallback as only tokenizer is available")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

            # Enhanced rule-based sentiment analysis with more comprehensive word lists
            # Negative words
            negative_words = [
                "दुःख", "पीडा", "रिस", "क्रोध", "भय", "चिन्ता", "नराम्रो", "खराब",
                "निराश", "कष्ट", "द्वेष", "घृणा", "दुर्भाग्य", "पीडित", "असफल", "बिरामी",
                "गरिब", "दुर्बल", "मृत्यु", "अस्वीकार", "हार", "टुटेको", "दुःखी", "रुन्",
                "कडा", "गाली", "आलोचना", "नकारात्मक", "विपत्ति", "खतरा", "समस्या", "अन्धकार", "चूर", "सकिनँ", "छैन"
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
            return {"sentiment": sentiment, "confidence": confidence, "rule_based": True,
                    "pos_matches": pos_matches, "neg_matches": neg_matches}

        # Full model is available
        # Load model and tokenizer
        model = SentimentClassifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()

        tokenizer_path = os.path.dirname(
            MODEL_PATH) if os.path.isfile(MODEL_PATH) else MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Tokenize input
        encoding = tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Map prediction to sentiment (0: Negative, 1: Neutral, 2: Positive)
        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        return {"sentiment": sentiment_map[predicted_class], "confidence": confidence}

    except Exception as e:
        print(f"Error in Nepali sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"sentiment": "Neutral", "confidence": 0.33, "error": str(e)}
