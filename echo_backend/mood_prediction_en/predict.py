import torch
from transformers import AutoTokenizer, AutoModel
import os
import re
from django.conf import settings

# Define model and tokenizer paths
MODEL_DIR = os.path.join(settings.BASE_DIR, 'bert_model', 'en_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model_20250309_181920.pt')


class SentimentClassifier(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=6):
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


def clean_text(text):
    """Clean and normalize English text."""
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters except letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def predict_sentiment_rule_based(text, tokenizer=None):
    """Rule-based sentiment analysis for English text with confidence scores."""
    # Clean the text
    cleaned_text = clean_text(text)

    # If no tokenizer is provided, use the default one
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Emotion lexicons - words associated with each emotion
    emotion_words = {
        "Sadness": [
            "sad", "depressed", "unhappy", "miserable", "heartbroken", "gloomy", "down",
            "disappointed", "sorrow", "despair", "grief", "regret", "lonely", "melancholy",
            "hopeless", "upset", "devastated", "hurt", "loss", "crying", "tears", "sob",
            "mourn", "weep", "blue", "broke", "empty", "abandoned"
        ],
        "Happiness": [
            "happy", "joy", "delighted", "pleased", "glad", "ecstatic", "thrilled", "excited",
            "cheerful", "good", "wonderful", "great", "amazing", "fantastic", "awesome",
            "content", "satisfied", "enjoy", "smile", "laugh", "amused", "lucky", "blessed",
            "celebrating", "celebration", "fun", "pleasant", "nice", "peaceful"
        ],
        "Love": [
            "love", "adore", "cherish", "affection", "beloved", "romance", "romantic", "loving",
            "passion", "crush", "fond", "care", "caring", "admire", "appreciate", "devotion",
            "heart", "tender", "intimate", "connection", "bond", "relationship", "together",
            "partner", "couple", "spouse", "dedicated", "loyal", "faithful"
        ],
        "Anger": [
            "angry", "mad", "furious", "annoyed", "frustrated", "rage", "hate", "outraged",
            "disgusted", "irritated", "enraged", "hostile", "bitter", "resentful", "offended",
            "upset", "provoked", "infuriated", "indignant", "livid", "fierce", "hatred",
            "wrath", "contempt", "temper", "storm", "fury", "conflict", "fight", "violent"
        ],
        "Fear": [
            "afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous",
            "panic", "horror", "terror", "dread", "shock", "alarmed", "paranoid", "threatened",
            "danger", "insecure", "phobia", "concern", "suspicious", "intimidated", "uneasy",
            "apprehensive", "scared", "creepy", "eerie", "startled", "fear"
        ],
        "Surprise": [
            "surprised", "shocked", "amazed", "astonished", "stunned", "unexpected", "wow",
            "speechless", "startled", "disbelief", "wonder", "awe", "bewildered", "unexpected",
            "revelation", "sudden", "incredible", "unbelievable", "unpredictable", "bizarre",
            "strange", "extraordinary", "remarkable", "phenomenal", "jaw-dropping"
        ]
    }

    # Negation words
    negation_words = ["not", "no", "never", "neither",
                      "nor", "hardly", "barely", "scarcely", "without"]

    # Neutral words
    neutral_words = ["is", "am", "are", "was", "were",
                     "be", "being", "been", "have", "has", "had"]

    # Tokenize the text
    tokens = tokenizer.tokenize(cleaned_text)
    filtered_tokens = [token for token in tokens if token not in neutral_words]

    # Find matches for each emotion
    emotion_matches = {}
    for emotion, words in emotion_words.items():
        matches = []
        for word in words:
            if word in cleaned_text:
                matches.append(word)

        for token in filtered_tokens:
            if token in words or any(w.startswith(token) for w in words):
                matches.append(token)

        emotion_matches[emotion] = list(set(matches))

    # Check for negation that might flip the sentiment
    has_negation = any(neg in cleaned_text for neg in negation_words)

    # Count matches for each emotion
    emotion_counts = {emotion: len(matches)
                      for emotion, matches in emotion_matches.items()}

    # Find dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    max_count = emotion_counts[dominant_emotion]

    # If there are no emotion words or all counts are 0, default to Neutral
    if max_count == 0:
        return {
            "sentiment": "Neutral",
            "confidence": 0.5,
            "rule_based": True,
            "matches": {}
        }

    # Calculate confidence based on the difference between the top emotion and others
    total_matches = sum(emotion_counts.values())
    confidence = 0.5 + min((max_count / total_matches) * 0.5, 0.45)

    # Handle negation
    if has_negation:
        # If negation is detected, reduce confidence
        confidence = max(0.5, confidence - 0.1)

        # In case of happiness or love with negation, consider sadness
        if dominant_emotion in ["Happiness", "Love"]:
            sad_count = emotion_counts["Sadness"]
            if sad_count > 0 or max_count <= 2:
                dominant_emotion = "Sadness"

    return {
        "sentiment": dominant_emotion,
        "confidence": confidence,
        "rule_based": True,
        "matches": emotion_matches
    }


def predict_sentiment(text):
    """
    Predict sentiment of English text using BERT model or rule-based approach if model fails.
    Returns one of: 'Sadness', 'Happiness', 'Love', 'Anger', 'Fear', 'Surprise'
    """
    try:
        if not os.path.exists(MODEL_PATH):
            print(
                f"Model file not found at {MODEL_PATH}, using rule-based analysis.")
            return predict_sentiment_rule_based(text)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load model and tokenizer
        model = SentimentClassifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        print("Model loaded successfully")
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Tokenize input
        encoding = tokenizer.encode_plus(
            text,
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

        # Map prediction to sentiment
        sentiment_map = {
            0: "Sadness",
            1: "Happiness",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise"
        }

        return {
            "sentiment": sentiment_map[predicted_class],
            "confidence": confidence,
            "rule_based": False
        }

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        print("Falling back to rule-based sentiment analysis.")
        # Fall back to rule-based approach
        return predict_sentiment_rule_based(text)
