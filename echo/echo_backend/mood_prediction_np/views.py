import torch
from transformers import AutoTokenizer
import os
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from django.utils import timezone

from .models import SentimentAnalysis, BERTModel
from .serializers import SentimentAnalysisSerializer, SentimentPredictionSerializer

from mood_prediction_np.utils import clean_nepali_text
from mood_prediction_np.management.commands.train_bert_model import SentimentClassifier


class SentimentAnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing sentiment analysis results."""
    queryset = SentimentAnalysis.objects.all().order_by('-created_at')
    serializer_class = SentimentAnalysisSerializer
    
    @action(detail=False, methods=['get'])
    def latest(self, request):
        """Get the latest sentiment analyses."""
        latest = self.get_queryset()[:10]
        serializer = self.get_serializer(latest, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def positive(self, request):
        """Get positive sentiment analyses."""
        positive = self.get_queryset().filter(sentiment=1)[:20]
        serializer = self.get_serializer(positive, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def negative(self, request):
        """Get negative sentiment analyses."""
        negative = self.get_queryset().filter(sentiment=-1)[:20]
        serializer = self.get_serializer(negative, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def neutral(self, request):
        """Get neutral sentiment analyses."""
        neutral = self.get_queryset().filter(sentiment=0)[:20]
        serializer = self.get_serializer(neutral, many=True)
        return Response(serializer.data)


@api_view(['POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def predict_sentiment(request):
    """Authenticated API endpoint to predict sentiment of Nepali text."""
    
    serializer = SentimentPredictionSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    text = serializer.validated_data['text']
    save_result = serializer.validated_data.get('save_result', True)
    
    # Clean the text
    cleaned_text = clean_nepali_text(text)
    
    # Get the active model
    try:
        active_model = BERTModel.objects.get(is_active=True)
    except BERTModel.DoesNotExist:
        return Response(
            {"error": "No active model found. Please train a model first."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Check if model file exists
    if not os.path.exists(active_model.model_path):
        return Response(
            {"error": "Model file not found. The model may have been moved or deleted."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Load the model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get model directory
    model_dir = os.path.dirname(active_model.model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load model
    model = SentimentClassifier("bert-base-multilingual-cased")
    model.load_state_dict(torch.load(active_model.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Tokenize text
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
        _, preds = torch.max(outputs, dim=1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()
    
    # Convert prediction from [0, 1, 2] to [-1, 0, 1]
    sentiment = preds.item() - 1
    
    sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    sentiment_text = sentiment_map[sentiment]
    
    # Save the result if requested
    if save_result:
        SentimentAnalysis.objects.create(
            text=text,
            cleaned_text=cleaned_text,
            sentiment=sentiment,
            confidence=confidence
        )
    
    return Response({
        "text": text,
        "cleaned_text": cleaned_text,
        "sentiment": sentiment,
        "sentiment_text": sentiment_text,
        "confidence": confidence,
        "timestamp": timezone.now()
    })
