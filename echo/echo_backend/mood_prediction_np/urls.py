

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views


router = DefaultRouter()
router.register(r'analyses', views.SentimentAnalysisViewSet)


urlpatterns = [
    path('', include(router.urls)), 
    path('predict/', views.predict_sentiment, name='predict-sentiment'),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]
