import os
import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from transformers import AutoModel, AutoTokenizer
from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from django.utils import timezone

from .models import SentimentAnalysis_en
from .serializers import SentimentAnalysisSerializer_en
from .predict import predict_sentiment

# Define model and tokenizer paths
MODEL_DIR = os.path.join(settings.BASE_DIR, 'bert_model', 'en_model')
MODEL_PATH = os.path.join(
    MODEL_DIR, 'model_20250309_151405.pt')  # Your trained model
TOKENIZER_PATH = os.path.join(
    MODEL_DIR, 'tokenizer.json')  # Your tokenizer file

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model class (same architecture used during training)


class SentimentClassifier(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=6):  # Adjust num_classes if needed
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


# Load trained model
model = SentimentClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


@csrf_exempt
def predict_sentiment(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text", "")

            if not text:
                return JsonResponse({"error": "No text provided"}, status=400)

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

            # Predict sentiment
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, token_type_ids)
                predicted_class = torch.argmax(outputs, dim=1).item()

            return JsonResponse({"sentiment": predicted_class})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


@api_view(['POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def predict_sentiment_api(request):
    """Authenticated API endpoint to predict sentiment of English text."""
    if 'text' not in request.data:
        return Response({
            'status': 'error',
            'message': 'Missing required parameter: text'
        }, status=status.HTTP_400_BAD_REQUEST)

    # Get text from request data
    text = request.data['text']
    save_result = request.data.get('save_result', True)

    # Predict sentiment
    try:
        result = predict_sentiment(text)
        sentiment = result.get('sentiment', 'Neutral')
        confidence = result.get('confidence', 0.5)
        rule_based = result.get('rule_based', False)

        # Map the BERT emotion to a simplified sentiment for frontend
        positive_emotions = ['Happiness', 'Love']
        negative_emotions = ['Sadness', 'Anger', 'Fear']

        if sentiment in positive_emotions:
            simplified_sentiment = 'Positive'
        elif sentiment in negative_emotions:
            simplified_sentiment = 'Negative'
        else:
            simplified_sentiment = 'Neutral'

        # Save the result if requested
        if save_result:
            # Get the numeric sentiment value
            sentiment_map = {
                'Sadness': 0,
                'Happiness': 1,
                'Love': 2,
                'Anger': 3,
                'Fear': 4,
                'Surprise': 5
            }
            sentiment_value = sentiment_map.get(sentiment, 0)

            SentimentAnalysis_en.objects.create(
                text=text,
                sentiment=sentiment_value,
                confidence=confidence
            )

        return Response({
            'status': 'success',
            'data': {
                'text': text,
                'detailed_sentiment': sentiment,
                'sentiment': simplified_sentiment,
                'confidence': confidence,
                'rule_based': rule_based,
                'timestamp': timezone.now()
            }
        })

    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Error predicting sentiment: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SentimentAnalysisViewSet_en(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing English sentiment analysis results."""
    queryset = SentimentAnalysis_en.objects.all().order_by('-created_at')
    serializer_class = SentimentAnalysisSerializer_en

    @action(detail=False, methods=['get'])
    def latest(self, request):
        """Get the latest sentiment analyses."""
        latest = self.get_queryset()[:10]
        serializer = self.get_serializer(latest, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def positive(self, request):
        """Get positive sentiment analyses (Happiness and Love)."""
        positive = self.get_queryset().filter(sentiment__in=[1, 2])[
            :20]  # Happiness and Love
        serializer = self.get_serializer(positive, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def negative(self, request):
        """Get negative sentiment analyses (Sadness, Anger, Fear)."""
        negative = self.get_queryset().filter(sentiment__in=[0, 3, 4])[
            :20]  # Sadness, Anger, Fear
        serializer = self.get_serializer(negative, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def neutral(self, request):
        """Get neutral sentiment analyses (Surprise)."""
        neutral = self.get_queryset().filter(sentiment=5)[:20]  # Surprise
        serializer = self.get_serializer(neutral, many=True)
        return Response(serializer.data)
