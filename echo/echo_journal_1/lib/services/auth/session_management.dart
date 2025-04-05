// lib/services/auth/session_manager.dart

import 'dart:async';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_fe/services/auth/secure_storage_service.dart';
import 'package:dio/dio.dart';
import 'package:echo_fe/core/configs/api_config.dart';
import 'package:flutter/foundation.dart';

/// 🔐 Service class handling JWT token management including validation,
/// refresh, and authentication state across app restarts.
class SessionManager {
  /// ⏱️ Duration in minutes before token refresh (based on JWT token lifetime)
  static const int tokenRefreshMinutes = 14; // Refresh 1 minute before expiry

  /// 📡 Callback function for authentication state changes
  static Function(bool)? _onAuthStateChanged;

  /// 🚀 Initialize auth manager and validate existing tokens silently
  /// Returns true if user is authenticated, false otherwise
  static Future<bool> initializeSilently() async {
    try {
      final token = await SecureStorageService.getAccessToken();
      if (token != null) {
        // Check token validity without making a network call
        final tokenInfo = await getTokenInfo();
        if (tokenInfo['isValid']) {
          return true;
        }
      }
      return false;
    } catch (e) {
      debugPrint('❌ Error initializing session: $e');
      return false;
    }
  }

  /// 🚀 Initialize auth manager and validate existing tokens
  ///
  /// Takes a callback function that will be called whenever auth state changes
  static Future<void> initialize(Function(bool) onAuthStateChanged) async {
    try {
      final isAuthenticated = await initializeSilently();
      onAuthStateChanged(isAuthenticated);
    } catch (e) {
      debugPrint('❌ Error initializing session: $e');
      onAuthStateChanged(false);
    }
  }

  /// ✨ Start a new authenticated session after login
  ///
  /// Updates login state in preferences
  static Future<void> startSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isLoggedIn', true);
    debugPrint('✅ Authentication successful');
  }

  /// 🔍 Check if current token is valid
  ///
  /// Returns true if token exists and hasn't expired
  static Future<bool> checkTokenValidity() async {
    // First check if we have a token in secure storage
    final token = await SecureStorageService.getAccessToken();

    if (token == null) {
      debugPrint('❌ No access token found');
      return false;
    }

    try {
      // Check token validity with the server
      await _verifyTokenWithServer(token);
      return true;
    } catch (e) {
      debugPrint('❌ Token validation error: $e');

      // Try to refresh the token
      final refreshToken = await SecureStorageService.getRefreshToken();
      if (refreshToken != null) {
        final success = await refreshAccessToken(refreshToken);
        if (success) {
          debugPrint('✅ Token refreshed successfully');
          return true;
        }
      }

      // If refresh failed or no refresh token, end the session
      await endSession();
      return false;
    }
  }

  /// 🚪 End current authentication and logout user
  ///
  /// Clears token data
  static Future<void> endSession() async {
    // Clear both secure storage and SharedPreferences
    await SecureStorageService.clearAuthData();

    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();

    debugPrint('✅ User logged out - tokens cleared');
  }

  /// 📊 Get current token information
  ///
  /// Returns a Map containing token status
  static Future<Map<String, dynamic>> getTokenInfo() async {
    final token = await SecureStorageService.getAccessToken();
    final loginTimestamp = await SecureStorageService.getLoginTimestamp();

    if (token == null || loginTimestamp == null) {
      return {'isValid': false, 'tokenAge': 0};
    }

    final currentTime = DateTime.now().millisecondsSinceEpoch;
    final tokenAge = (currentTime - loginTimestamp) / (1000 * 60);

    return {
      'isValid': tokenAge < 60 * 24, // 24 hours - max reasonable token lifetime
      'tokenAge': tokenAge.round(),
    };
  }

  static Future<bool> checkLoginStatus() async {
    return await SecureStorageService.isLoggedIn();
  }

  static Future<String?> getAccessToken() async {
    return await SecureStorageService.getAccessToken();
  }

  static Future<String?> getRefreshToken() async {
    return await SecureStorageService.getRefreshToken();
  }

  static Future<Map<String, String?>> getUserData() async {
    return await SecureStorageService.getUserData();
  }

  /// Attempts to refresh the access token using the refresh token
  static Future<bool> refreshAccessToken(String refreshToken) async {
    try {
      final dio = Dio();
      final url = ApiConfig.getFullUrl(ApiConfig.tokenRefreshEndpoint);
      final response = await dio.post(
        url,
        data: {'refresh': refreshToken},
      );

      if (response.statusCode == 200 && response.data['access'] != null) {
        // Save the new access token
        await SecureStorageService.saveAuthData(
          accessToken: response.data['access'],
          refreshToken: refreshToken,
          userId: (await SecureStorageService.getUserId()) ?? '',
          username: (await SecureStorageService.getUsername()) ?? '',
          email: (await SecureStorageService.getUserEmail()) ?? '',
        );

        debugPrint('✅ Access token refreshed successfully');
        return true;
      }

      debugPrint('❌ Token refresh failed: ${response.statusCode}');
      return false;
    } catch (e) {
      debugPrint('❌ Token refresh error: $e');
      return false;
    }
  }

  /// Makes a lightweight API call to verify the token with the server
  static Future<void> _verifyTokenWithServer(String token) async {
    try {
      final dio = Dio();
      debugPrint('🔍 Verifying token with server...');

      final url = ApiConfig.getFullUrl(ApiConfig.tokenVerifyEndpoint);
      final response = await dio.post(
        url,
        options: Options(
          headers: {'Authorization': 'Bearer $token'},
          validateStatus: (status) => true,
        ),
      );

      if (response.statusCode != 200) {
        debugPrint('❌ Token verification failed: ${response.statusCode}');
        await SecureStorageService.clearAuthData();
      } else {
        debugPrint('✅ Token verified successfully');
      }
    } catch (e) {
      debugPrint('❌ Error verifying token: $e');
      await SecureStorageService.clearAuthData();
    }
  }
}
