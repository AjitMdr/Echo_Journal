import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_journal_1/core/configs/api_config.dart';
import '../../models/subscription/plan.dart';
import '../../models/subscription/subscription.dart';
import '../../models/subscription/payment.dart';

class SubscriptionService {
  final String authPrefix = '/auth';

  Future<String?> _getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('token');
  }

  Future<List<Plan>> getPlans() async {
    final token = await _getToken();
    if (token == null) {
      throw Exception('No authentication token found');
    }

    final url = ApiConfig.getFullUrl('$authPrefix/subscription/plans/');
    final response = await http.get(
      Uri.parse(url),
      headers: {
        'Authorization': 'Token $token',
        'Content-Type': 'application/json',
      },
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((plan) => Plan.fromJson(plan)).toList();
    } else {
      throw Exception('Failed to load plans');
    }
  }

  Future<Subscription> getCurrentSubscription() async {
    final token = await _getToken();
    if (token == null) {
      throw Exception('No authentication token found');
    }

    final url = ApiConfig.getFullUrl('$authPrefix/subscription/subscriptions/');
    final response = await http.get(
      Uri.parse(url),
      headers: {
        'Authorization': 'Token $token',
        'Content-Type': 'application/json',
      },
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      if (data.isEmpty) {
        throw Exception('No active subscription found');
      }
      return Subscription.fromJson(data[0]);
    } else {
      throw Exception('Failed to load subscription');
    }
  }

  Future<Subscription> subscribe(int planId) async {
    final token = await _getToken();
    if (token == null) {
      throw Exception('No authentication token found');
    }

    final url = ApiConfig.getFullUrl('$authPrefix/subscription/subscriptions/');
    final response = await http.post(
      Uri.parse(url),
      headers: {
        'Authorization': 'Token $token',
        'Content-Type': 'application/json',
      },
      body: json.encode({'plan': planId}),
    );

    if (response.statusCode == 201) {
      return Subscription.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to create subscription');
    }
  }

  Future<void> cancelSubscription() async {
    final token = await _getToken();
    if (token == null) {
      throw Exception('No authentication token found');
    }

    final url = ApiConfig.getFullUrl('$authPrefix/subscription/cancel/');
    final response = await http.post(
      Uri.parse(url),
      headers: {
        'Authorization': 'Token $token',
        'Content-Type': 'application/json',
      },
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to cancel subscription');
    }
  }

  Future<Payment> createPayment(String planId) async {
    final token = await _getToken();
    if (token == null) {
      throw Exception('No authentication token found');
    }

    final url = ApiConfig.getFullUrl('$authPrefix/subscription/payments/');
    final response = await http.post(
      Uri.parse(url),
      headers: {
        'Authorization': 'Token $token',
        'Content-Type': 'application/json',
      },
      body: json.encode({'plan_id': planId}),
    );

    if (response.statusCode == 201) {
      return Payment.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to create payment');
    }
  }

  Future<List<Payment>> getPaymentHistory() async {
    final token = await _getToken();
    if (token == null) {
      throw Exception('No authentication token found');
    }

    final url = ApiConfig.getFullUrl('$authPrefix/subscription/payments/');
    final response = await http.get(
      Uri.parse(url),
      headers: {
        'Authorization': 'Token $token',
        'Content-Type': 'application/json',
      },
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((payment) => Payment.fromJson(payment)).toList();
    } else {
      throw Exception('Failed to load payment history');
    }
  }
}
