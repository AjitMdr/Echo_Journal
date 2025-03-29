from rest_framework import serializers
from .models import SentimentAnalysis, BERTModel

class SentimentAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for the SentimentAnalysis model."""
    
    sentiment_display = serializers.SerializerMethodField()
    
    class Meta:
        model = SentimentAnalysis
        fields = ['id', 'text', 'cleaned_text', 'sentiment', 'sentiment_display', 'confidence', 'created_at']
        read_only_fields = ['id', 'cleaned_text', 'sentiment', 'sentiment_display', 'confidence', 'created_at']
    
    def get_sentiment_display(self, obj):
        sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        return sentiment_map.get(obj.sentiment, "Unknown")

class BERTModelSerializer(serializers.ModelSerializer):
    """Serializer for the BERTModel."""
    
    class Meta:
        model = BERTModel
        fields = ['id', 'name', 'description', 'accuracy', 'f1_score', 'training_completed', 'is_active']
        read_only_fields = ['id', 'accuracy', 'f1_score', 'training_completed']

class SentimentPredictionSerializer(serializers.Serializer):
    """Serializer for sentiment prediction requests."""
    
    text = serializers.CharField(required=True, help_text="Nepali text for sentiment analysis")
    save_result = serializers.BooleanField(required=False, default=True, 
                                          help_text="Whether to save the analysis result to the database")
    
    def validate_text(self, value):
        if len(value.strip()) < 5:
            raise serializers.ValidationError("Text must be at least 5 characters long")
        return value