from django.db import models
from django.utils import timezone

class SentimentAnalysis_en(models.Model):
    """Model to store sentiment analysis results for English text."""

    SENTIMENT_CHOICES = [
        (0, 'Sadness'),
        (1, 'Happiness'),
        (2, 'Love'),
        (3, 'Anger'),
        (4, 'Fear'),
        (5, 'Surprise'),
    ]

    text = models.TextField(verbose_name="Original Text")
    sentiment = models.IntegerField(choices=SENTIMENT_CHOICES, verbose_name="Sentiment")
    confidence = models.FloatField(verbose_name="Confidence Score")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        verbose_name = "English Sentiment Analysis"
        verbose_name_plural = "English Sentiment Analyses"
        ordering = ['-created_at']

    def __str__(self):
        sentiment_map = {
            0: "Sadness",
            1: "Happiness",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise"
        }
        return f"{self.text[:50]}... | {sentiment_map[self.sentiment]} ({self.confidence:.2f})"


class BERTModel(models.Model):
    """Model to keep track of trained BERT models for English sentiment analysis."""

    name = models.CharField(max_length=255, verbose_name="Model Name")
    description = models.TextField(blank=True, null=True, verbose_name="Description")
    model_path = models.CharField(max_length=255, verbose_name="Model Path")
    accuracy = models.FloatField(verbose_name="Accuracy Score")
    f1_score = models.FloatField(verbose_name="F1 Score")
    training_completed = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=False, verbose_name="Active Model")

    class Meta:
        verbose_name = "English BERT Model"
        verbose_name_plural = "English BERT Models"
        ordering = ['-training_completed']

    def __str__(self):
        return f"{self.name} ({self.accuracy:.2f} acc, {self.f1_score:.2f} F1)"

    def save(self, *args, **kwargs):
        # Ensure only one model is active at a time
        if self.is_active:
            BERTModel.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)
