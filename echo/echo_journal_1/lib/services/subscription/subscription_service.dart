import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:echo_journal1/core/configs/api_config.dart';
import '../../models/subscription/plan.dart';
import '../../models/subscription/subscription.dart';
import '../../models/subscription/payment.dart';
import '../../services/auth/secure_storage_service.dart';
import 'package:flutter/foundation.dart';

class SubscriptionService {
  Future<String?> _getToken() async {
    return await SecureStorageService.getAccessToken();
  }

  Map<String, String> _getAuthHeaders(String token) {
    return {
      'Authorization': 'Bearer $token',
      'Content-Type': 'application/json',
    };
  }

  Future<List<Plan>> getPlans() async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('Please log in to view subscription plans');
      }

      final url = ApiConfig.getFullUrl('subscription/plans');
      debugPrint('🔍 Fetching plans from: $url');

      final response = await http.get(
        Uri.parse(url),
        headers: _getAuthHeaders(token),
      );

      debugPrint('📡 Response status: ${response.statusCode}');
      debugPrint('📡 Response body: ${response.body}');

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        return data.map((plan) => Plan.fromJson(plan)).toList();
      } else if (response.statusCode == 401) {
        throw Exception('Session expired. Please log in again');
      } else if (response.statusCode == 404) {
        throw Exception(
            'Subscription plans endpoint not found. Please check your backend server is running and accessible.');
      } else if (response.statusCode == 500) {
        throw Exception(
            'Server error. Please try again later or contact support.');
      } else {
        String errorMessage = 'Failed to load plans';
        try {
          final errorData = json.decode(response.body);
          if (errorData['detail'] != null) {
            errorMessage = errorData['detail'];
          }
        } catch (e) {
          // If we can't parse the error message, just use the status code
          errorMessage = '$errorMessage: ${response.statusCode}';
        }
        throw Exception(errorMessage);
      }
    } catch (e) {
      debugPrint('❌ Error fetching plans: $e');
      throw Exception(e.toString().replaceAll('Exception: ', ''));
    }
  }

  Future<Subscription> getCurrentSubscription() async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('Please log in to view your subscription');
      }

      final url = ApiConfig.getFullUrl('subscription/subscriptions');
      final response = await http.get(
        Uri.parse(url),
        headers: _getAuthHeaders(token),
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        if (data.isEmpty) {
          return await _createFreeSubscription();
        }

        // Convert all subscriptions to Subscription objects
        final subscriptions =
            data.map((s) => Subscription.fromJson(s)).toList();

        // First, look for an active premium subscription
        final activePremium = subscriptions.firstWhere(
          (s) => s.status == 'ACTIVE' && s.planDetails?.planType == 'PREMIUM',
          orElse: () => Subscription.fromJson({}),
        );

        if (activePremium.id != 0) {
          return activePremium;
        }

        // If no active premium, look for an active free subscription
        final activeFree = subscriptions.firstWhere(
          (s) => s.status == 'ACTIVE' && s.planDetails?.planType == 'FREE',
          orElse: () => Subscription.fromJson({}),
        );

        if (activeFree.id != 0) {
          return activeFree;
        }

        // If no active subscription found, create a new free subscription
        return await _createFreeSubscription();
      } else if (response.statusCode == 401) {
        throw Exception('Session expired. Please log in again');
      } else {
        throw Exception('Failed to load subscription: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString().replaceAll('Exception: ', ''));
    }
  }

  Future<Subscription> _createFreeSubscription() async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('Please log in to create a subscription');
      }

      // Get the free plan
      final plans = await getPlans();
      final freePlan = plans.firstWhere(
        (plan) => plan.planType == 'FREE',
        orElse: () => throw Exception('Free plan not found'),
      );

      final url = ApiConfig.getFullUrl('subscription/subscriptions');
      final response = await http.post(
        Uri.parse(url),
        headers: _getAuthHeaders(token),
        body: json.encode({'plan': freePlan.id}),
      );

      if (response.statusCode == 201) {
        return Subscription.fromJson(json.decode(response.body));
      } else if (response.statusCode == 401) {
        throw Exception('Session expired. Please log in again');
      } else {
        throw Exception(
            'Failed to create subscription: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString().replaceAll('Exception: ', ''));
    }
  }

  Future<Subscription> subscribe(int planId) async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('Please log in to subscribe');
      }

      final url = ApiConfig.getFullUrl('subscription/subscriptions');
      final response = await http.post(
        Uri.parse(url),
        headers: _getAuthHeaders(token),
        body: json.encode({'plan': planId, 'expire_existing': true}),
      );

      if (response.statusCode == 201) {
        return Subscription.fromJson(json.decode(response.body));
      } else if (response.statusCode == 401) {
        throw Exception('Session expired. Please log in again');
      } else {
        throw Exception(
            'Failed to create subscription: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString().replaceAll('Exception: ', ''));
    }
  }

  Future<void> cancelSubscription() async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('Please log in to cancel subscription');
      }

      final url = ApiConfig.getFullUrl('subscription/subscriptions/cancel');
      final response = await http.post(
        Uri.parse(url),
        headers: _getAuthHeaders(token),
      );

      if (response.statusCode == 401) {
        throw Exception('Session expired. Please log in again');
      } else if (response.statusCode != 200) {
        throw Exception(
            'Failed to cancel subscription: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString().replaceAll('Exception: ', ''));
    }
  }

  Future<Payment> createPayment(String planId, {required String refId}) async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('Please log in to make a payment');
      }

      // Get the plan to access its price
      final plans = await getPlans();
      final plan = plans.firstWhere(
        (p) => p.id == int.parse(planId),
        orElse: () => throw Exception('Plan not found'),
      );

      final url = ApiConfig.getFullUrl('subscription/payments');
      final response = await http.post(
        Uri.parse(url),
        headers: _getAuthHeaders(token),
        body: json.encode({
          'plan': int.parse(planId),
          'payment_method': 'ESEWA',
          'transaction_id': refId,
          'status': 'SUCCESS',
          'amount': plan.price,
          'currency': 'NPR',
          'create_subscription':
              true, // Add flag to create subscription in same request
          'expire_existing': true // Add flag to expire existing subscriptions
        }),
      );

      if (response.statusCode == 201) {
        return Payment.fromJson(json.decode(response.body));
      } else if (response.statusCode == 401) {
        throw Exception('Session expired. Please log in again');
      } else {
        String errorMessage = 'Failed to create payment';
        try {
          final errorData = json.decode(response.body);
          if (errorData['detail'] != null) {
            errorMessage = errorData['detail'];
          }
        } catch (e) {
          errorMessage = '$errorMessage: ${response.statusCode}';
        }
        throw Exception(errorMessage);
      }
    } catch (e) {
      throw Exception(e.toString().replaceAll('Exception: ', ''));
    }
  }

  Future<double> _getPlanPrice(int planId) async {
    try {
      final plans = await getPlans();
      final plan = plans.firstWhere(
        (p) => p.id == planId,
        orElse: () => throw Exception('Plan not found'),
      );
      return plan.price;
    } catch (e) {
      throw Exception('Failed to get plan price: ${e.toString()}');
    }
  }

  Future<List<Payment>> getPaymentHistory() async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('Please log in to view payment history');
      }

      final url = ApiConfig.getFullUrl('subscription/payments');
      final response = await http.get(
        Uri.parse(url),
        headers: _getAuthHeaders(token),
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        return data.map((payment) => Payment.fromJson(payment)).toList();
      } else if (response.statusCode == 401) {
        throw Exception('Session expired. Please log in again');
      } else {
        throw Exception(
            'Failed to load payment history: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString().replaceAll('Exception: ', ''));
    }
  }
}
