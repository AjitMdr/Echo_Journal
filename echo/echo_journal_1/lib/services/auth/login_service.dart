import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';
import 'package:flutter/foundation.dart';
import 'package:echo_journal1/services/auth/session_management.dart';
import 'package:echo_journal1/core/configs/api_config.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class AuthService {
  /// 🌐 HTTP client instance with default configuration
  static final Dio dio = Dio(
    BaseOptions(
      connectTimeout: Duration(seconds: 10),
      receiveTimeout: Duration(seconds: 10),
      validateStatus: (status) => status != null && status! < 500,
      contentType: 'application/json',
      headers: {
        'Accept': 'application/json',
      },
      followRedirects: false,
    ),
  );

  static final String _authPrefix = '/auth';

  // Initialize Dio with interceptors
  static void initializeDio() {
    dio.interceptors.add(
      InterceptorsWrapper(
        onError: (DioException e, handler) {
          debugPrint('🚨 Auth DioError: ${e.type} - ${e.message}');
          debugPrint('🚨 Request: ${e.requestOptions.uri}');
          if (e.response != null) {
            debugPrint(
                '🚨 Response: ${e.response?.statusCode} - ${e.response?.data}');

            // Check if response is HTML
            final contentType = e.response?.headers.value('content-type') ?? '';
            if (contentType.toLowerCase().contains('text/html')) {
              // Create a new error response with our custom message
              e.response?.data = {
                'detail': 'Server error occurred. Please try again later.'
              };
            }
          }
          return handler.next(e);
        },
      ),
    );
    debugPrint('✅ Auth service Dio interceptors initialized');
  }

  // Call this method during app initialization
  static void initialize() {
    initializeDio();
  }

  /// Authenticates user and securely stores the JWT tokens if successful
  static Future<Map<String, dynamic>> login(
      String username, String password) async {
    try {
      final url = ApiConfig.getFullUrl('$_authPrefix/login/');
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'username': username,
          'password': password,
        }),
      );

      final data = json.decode(response.body);

      if (response.statusCode == 200) {
        // Check if 2FA is required
        if (data['requires_2fa'] == true) {
          return {
            'requires_2fa': true,
            'email': data['email'],
            'message': data['message'],
          };
        }

        // Save complete auth data for non-2FA users
        await SecureStorageService.saveAuthData(
          accessToken: data['access'],
          refreshToken: data['refresh'],
          userId: data['user']['id'].toString(),
          username: data['user']['username'],
          email: data['user']['email'],
        );

        // Return the original response
        return data;
      } else {
        return {
          'success': false,
          'error': data['error'],
          'needs_verification': data['needs_verification'] ?? false,
          'email': data['email'],
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': e.toString(),
      };
    }
  }

  static Future<Map<String, dynamic>> verify2FALogin(
      String email, String otp) async {
    try {
      final url = ApiConfig.getFullUrl('$_authPrefix/login/2fa/verify/');
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
          'otp': otp,
        }),
      );

      final data = json.decode(response.body);

      if (response.statusCode == 200) {
        // Save complete auth data including user information
        await SecureStorageService.saveAuthData(
          accessToken: data['access'],
          refreshToken: data['refresh'],
          userId: data['user']['id'].toString(),
          username: data['user']['username'],
          email: data['user']['email'],
        );

        return {
          'success': true,
          'user': data['user'],
        };
      } else {
        return {
          'success': false,
          'error': data['error'],
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': e.toString(),
      };
    }
  }

  static Future<Map<String, dynamic>> resend2FACode(String email) async {
    try {
      final url = ApiConfig.getFullUrl('$_authPrefix/login/2fa/resend/');
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
        }),
      );

      final data = json.decode(response.body);

      if (response.statusCode == 200) {
        return {
          'success': true,
          'message': data['message'],
        };
      } else {
        return {
          'success': false,
          'error': data['error'],
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': e.toString(),
      };
    }
  }

  /// Save authentication tokens securely
  static Future<void> saveTokens(
      String accessToken, String refreshToken) async {
    await SecureStorageService.setAccessToken(accessToken);
    await SecureStorageService.setRefreshToken(refreshToken);
  }

  /// 🚪 Logs out the current user by clearing all tokens and session data
  static Future<void> logout() async {
    debugPrint('🚪 Starting logout process...');

    try {
      // End the session first (this will clear secure storage)
      await SessionManager.endSession();

      // Clear SharedPreferences
      final prefs = await SharedPreferences.getInstance();
      await prefs.clear();

      // Double check to ensure all secure storage is cleared
      await SecureStorageService.clearAuthData();

      // Verify all data is cleared by checking storage
      final accessToken = await SecureStorageService.getAccessToken();
      final refreshToken = await SecureStorageService.getRefreshToken();
      final userId = await SecureStorageService.getUserId();

      if (accessToken == null && refreshToken == null && userId == null) {
        debugPrint('✅ All authentication data successfully cleared');
      } else {
        debugPrint('⚠️ Warning: Some auth data may still remain');
        // Force clear again if any data remains
        await SecureStorageService.clearAuthData();
      }
    } catch (e) {
      debugPrint('❌ Error during logout: ${e.toString()}');
      // Ensure secure storage is cleared even if other operations fail
      await SecureStorageService.clearAuthData();
    }
  }

  /// 🔒 Check if user is currently authenticated
  static Future<bool> isLoggedIn() async {
    return await SecureStorageService.isLoggedIn();
  }

  /// 🔑 Get the current access token securely
  static Future<String?> getToken() async {
    return await SecureStorageService.getAccessToken();
  }
}
