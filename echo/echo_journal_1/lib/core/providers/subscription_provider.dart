import 'package:flutter/material.dart';
import '../../models/subscription/subscription.dart';
import '../../services/subscription/subscription_service.dart';
import '../../services/auth/secure_storage_service.dart';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class SubscriptionProvider extends ChangeNotifier {
  final SubscriptionService _subscriptionService = SubscriptionService();
  Subscription? _subscription;
  bool _isLoading = true;
  String? _error;
  static const String _cacheKey = 'subscription_cache';
  static const Duration _cacheDuration = Duration(minutes: 5);
  DateTime? _lastChecked;

  bool get isLoading => _isLoading;
  String? get error => _error;
  Subscription? get subscription => _subscription;
  bool get isPremium => _subscription?.isPremium ?? false;
  bool get isFree => _subscription?.isFree ?? false;

  SubscriptionProvider() {
    _initializeFromCache();
  }

  Future<void> _initializeFromCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final cachedData = prefs.getString(_cacheKey);

      if (cachedData != null) {
        final data = json.decode(cachedData);
        final lastChecked = DateTime.parse(data['timestamp']);

        if (DateTime.now().difference(lastChecked) < _cacheDuration) {
          _subscription = Subscription.fromJson(data['subscription']);
          _lastChecked = lastChecked;
          _isLoading = false;
          _error = null;
          notifyListeners();
          return;
        }
      }

      await checkSubscription();
    } catch (e) {
      // If there's an error reading cache, proceed with normal subscription check
      await checkSubscription();
    }
  }

  Future<void> _updateCache() async {
    try {
      if (_subscription != null) {
        final prefs = await SharedPreferences.getInstance();
        final data = {
          'subscription': _subscription!.toJson(),
          'timestamp': DateTime.now().toIso8601String(),
        };
        await prefs.setString(_cacheKey, json.encode(data));
      }
    } catch (e) {
      // Cache update failure shouldn't block the app
      debugPrint('Failed to update subscription cache: $e');
    }
  }

  Future<void> checkSubscription() async {
    try {
      _isLoading = true;
      _error = null;
      notifyListeners();

      // Check if token exists
      final token = await SecureStorageService.getToken();
      if (token == null) {
        throw Exception('Please log in to access this feature');
      }

      final subscription = await _subscriptionService.getCurrentSubscription();
      _subscription = subscription;
      _lastChecked = DateTime.now();
      _error = null;
      await _updateCache();
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> refreshSubscription() async {
    // Clear cache and check subscription
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_cacheKey);
    } catch (e) {
      debugPrint('Failed to clear subscription cache: $e');
    }
    await checkSubscription();
  }
}
