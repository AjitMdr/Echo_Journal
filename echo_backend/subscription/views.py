from django.shortcuts import render
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from .models import Plan, Subscription, Payment
from .serializers import (
    PlanSerializer,
    SubscriptionSerializer,
    PaymentSerializer
)


class PlanViewSet(viewsets.ModelViewSet):
    """ViewSet for subscription plans"""
    queryset = Plan.objects.filter(is_active=True)
    serializer_class = PlanSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_permissions(self):
        """Set custom permissions for different actions"""
        if self.action in ['list', 'retrieve']:
            return [permissions.IsAuthenticated()]
        return [permissions.IsAdminUser()]


class SubscriptionViewSet(viewsets.ModelViewSet):
    """ViewSet for user subscriptions"""
    serializer_class = SubscriptionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Filter subscriptions by user"""
        return Subscription.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        """Create a new subscription"""
        plan = serializer.validated_data['plan']
        start_date = timezone.now()
        end_date = start_date + timezone.timedelta(days=plan.duration_days)
        
        serializer.save(
            user=self.request.user,
            start_date=start_date,
            end_date=end_date,
            status='ACTIVE'
        )

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a subscription"""
        subscription = self.get_object()
        subscription.cancel()
        return Response(
            {'status': 'Subscription cancelled'},
            status=status.HTTP_200_OK
        )

    @action(detail=True, methods=['post'])
    def renew(self, request, pk=None):
        """Renew a subscription"""
        subscription = self.get_object()
        subscription.renew()
        serializer = self.get_serializer(subscription)
        return Response(serializer.data)


class PaymentViewSet(viewsets.ModelViewSet):
    """ViewSet for subscription payments"""
    serializer_class = PaymentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Filter payments by user's subscriptions"""
        return Payment.objects.filter(
            subscription__user=self.request.user
        )

    def perform_create(self, serializer):
        """Create a new payment"""
        # Generate a unique transaction ID (in practice, use a payment gateway)
        import uuid
        transaction_id = str(uuid.uuid4())
        
        serializer.save(
            transaction_id=transaction_id,
            status='SUCCESS'  # In practice, this would depend on payment gateway
        )
