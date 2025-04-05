from rest_framework import serializers
from mood_prediction_en.models import SentimentAnalysis_en, BERTModel

class SentimentAnalysisSerializer_en(serializers.ModelSerializer):
    """Serializer for the SentimentAnalysis model."""

    sentiment_text = serializers.SerializerMethodField()

    class Meta:
        model = SentimentAnalysis_en
        fields = ['id', 'text', 'sentiment', 'sentiment_text', 'confidence', 'created_at']

    def get_sentiment_text(self, obj):
        """Convert sentiment label to human-readable text."""
        sentiment_map = {
            0: "Sadness",
            1: "Happiness",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise"
        }
        return sentiment_map.get(obj.sentiment, "Unknown")


class SentimentPredictionSerializer_en(serializers.Serializer):
    """Serializer for handling sentiment prediction requests."""
    
    text = serializers.CharField()
    save_result = serializers.BooleanField(default=True)


class BERTModelSerializer(serializers.ModelSerializer):
    """Serializer for the BERTModel model."""

    class Meta:
        model = BERTModel
        fields = ['id', 'name', 'description', 'model_path', 'accuracy', 'f1_score', 'training_completed', 'is_active']
