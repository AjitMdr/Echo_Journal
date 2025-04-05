from django.urls import path, include
from rest_framework.routers import DefaultRouter
from mood_prediction_en.views import SentimentAnalysisViewSet_en, predict_sentiment_en

# Router for SentimentAnalysis ViewSet
router = DefaultRouter()
router.register(r'sentiments', SentimentAnalysisViewSet_en, basename='sentiments')

urlpatterns = [
    path('api/', include(router.urls)),  
    path('predict/', predict_sentiment_en, name='predict_sentiment'),  # Predict API
]
