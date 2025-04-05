from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PlanViewSet, SubscriptionViewSet, PaymentViewSet

app_name = 'subscription'

router = DefaultRouter()
router.register('plans', PlanViewSet, basename='plan')
router.register('subscriptions', SubscriptionViewSet, basename='subscription')
router.register('payments', PaymentViewSet, basename='payment')

urlpatterns = [
    path('', include(router.urls)),
]
