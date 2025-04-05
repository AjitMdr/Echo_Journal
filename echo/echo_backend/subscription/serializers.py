from rest_framework import serializers
from .models import Plan, Subscription, Payment


class PlanSerializer(serializers.ModelSerializer):
    """Serializer for subscription plans"""
    class Meta:
        model = Plan
        fields = [
            'id', 'name', 'plan_type', 'price', 'duration_days',
            'description', 'features', 'is_active', 'created_at',
            'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class SubscriptionSerializer(serializers.ModelSerializer):
    """Serializer for user subscriptions"""
    plan_details = PlanSerializer(source='plan', read_only=True)
    days_remaining = serializers.SerializerMethodField()

    class Meta:
        model = Subscription
        fields = [
            'id', 'user', 'plan', 'plan_details', 'status',
            'start_date', 'end_date', 'is_auto_renewal',
            'created_at', 'updated_at', 'days_remaining'
        ]
        read_only_fields = ['user', 'status', 'start_date', 'end_date',
                            'created_at', 'updated_at', 'days_remaining']

    def get_days_remaining(self, obj):
        """Calculate days remaining in subscription"""
        if obj.is_active:
            from django.utils import timezone
            now = timezone.now()
            remaining = obj.end_date - now
            return max(0, remaining.days)
        return 0


class PaymentSerializer(serializers.ModelSerializer):
    """Serializer for subscription payments"""
    subscription_details = SubscriptionSerializer(
        source='subscription',
        read_only=True
    )

    class Meta:
        model = Payment
        fields = [
            'id', 'subscription', 'subscription_details',
            'amount', 'currency', 'payment_method', 'status',
            'transaction_id', 'payment_date', 'last_modified'
        ]
        read_only_fields = [
            'id', 'payment_date', 'last_modified'
        ]
        extra_kwargs = {
            # Make subscription optional initially
            'subscription': {'required': False},
            # Ensure transaction_id is provided
            'transaction_id': {'required': True}
        }

    def validate_amount(self, value):
        """Validate payment amount"""
        if value <= 0:
            raise serializers.ValidationError(
                "Payment amount must be greater than zero."
            )
        return value

    def validate_transaction_id(self, value):
        """Validate transaction ID"""
        if Payment.objects.filter(transaction_id=value).exists():
            raise serializers.ValidationError(
                "This transaction has already been processed."
            )
        return value
