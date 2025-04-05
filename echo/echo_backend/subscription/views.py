from django.shortcuts import render
from rest_framework import viewsets, status, permissions, serializers
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from datetime import timedelta
from .models import Plan, Subscription, Payment
from .serializers import (
    PlanSerializer,
    SubscriptionSerializer,
    PaymentSerializer
)
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from rest_framework_simplejwt.authentication import JWTAuthentication


class PlanViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing subscription plans"""
    queryset = Plan.objects.filter(is_active=True)
    serializer_class = PlanSerializer
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def list(self, request, *args, **kwargs):
        """List all active plans"""
        try:
            plans = self.get_queryset()
            serializer = self.get_serializer(plans, many=True)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SubscriptionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing user subscriptions"""
    serializer_class = SubscriptionSerializer
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        """Get subscriptions for the current user"""
        return Subscription.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        """Create a new subscription"""
        try:
            plan = serializer.validated_data['plan']
            end_date = timezone.now() + timedelta(days=plan.duration_days)
            serializer.save(
                user=self.request.user,
                status='ACTIVE',
                start_date=timezone.now(),
                end_date=end_date
            )
        except Exception as e:
            raise serializers.ValidationError(str(e))

    @action(detail=False, methods=['post'])
    def cancel(self, request):
        """Cancel the current active subscription"""
        try:
            subscription = self.get_queryset().get(status='ACTIVE')
            subscription.status = 'CANCELLED'
            subscription.save()
            return Response({'status': 'subscription cancelled'})
        except Subscription.DoesNotExist:
            return Response(
                {'error': 'No active subscription found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
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
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        """Filter payments by user's subscriptions"""
        return Payment.objects.filter(
            subscription__user=self.request.user
        ).order_by('-payment_date')

    def perform_create(self, serializer):
        """Create a new payment and subscription record"""
        try:
            # Get the plan ID from the request data
            plan_id = self.request.data.get('plan')
            if not plan_id:
                raise serializers.ValidationError("Plan ID is required")

            # Get the plan
            try:
                plan = Plan.objects.get(id=plan_id)
            except Plan.DoesNotExist:
                raise serializers.ValidationError("Invalid plan ID")

            # Expire all active subscriptions for this user
            Subscription.objects.filter(
                user=self.request.user,
                status='ACTIVE'
            ).update(
                status='EXPIRED',
                end_date=timezone.now(),
                updated_at=timezone.now()
            )

            # Create new subscription
            subscription = Subscription.objects.create(
                user=self.request.user,
                plan=plan,
                status='ACTIVE',
                start_date=timezone.now(),
                end_date=timezone.now() + timedelta(days=plan.duration_days),
                is_auto_renewal=False
            )

            # Create payment record
            payment = serializer.save(
                subscription=subscription,
                transaction_id=self.request.data.get('transaction_id'),
                payment_method='ESEWA',
                amount=plan.price,
                status='SUCCESS',
                currency='NPR'
            )

            return payment

        except Exception as e:
            raise serializers.ValidationError(str(e))
