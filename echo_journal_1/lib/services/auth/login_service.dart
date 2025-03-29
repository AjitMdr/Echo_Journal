import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_journal_1/services/auth/secure_storage_service.dart';
import 'package:flutter/foundation.dart';
import 'package:echo_journal_1/services/auth/session_management.dart';
import 'package:echo_journal_1/core/configs/api_config.dart';

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
    String username,
    String password,
  ) async {
    try {
      final url = ApiConfig.getFullUrl(ApiConfig.tokenEndpoint);
      debugPrint('🔐 Attempting login for user: $username');
      debugPrint('🌐 Login URL: $url');

      final response = await dio.post(
        url,
        data: {
          'username': username,
          'password': password,
        },
        options: Options(
          validateStatus: (status) => status != null && status! < 500,
          followRedirects: false,
          headers: {'Accept': 'application/json'},
        ),
      );

      debugPrint('📡 Login Response Status: ${response.statusCode}');

      // Check if response is HTML (indicates server error)
      final contentType = response.headers.value('content-type') ?? '';
      if (contentType.toLowerCase().contains('text/html')) {
        debugPrint('❌ Received HTML response instead of JSON');
        return {
          'success': false,
          'error': 'Server error occurred. Please try again later.',
        };
      }

      debugPrint('📡 Login Response Data: ${response.data}');

      if (response.statusCode == 200) {
        final data = response.data;
        if (data['access'] != null && data['refresh'] != null) {
          // Get user ID directly from the response
          String? userId;
          if (data['user'] != null && data['user']['id'] != null) {
            userId = data['user']['id'].toString();
            debugPrint('✅ Got user ID directly from response: $userId');
          }

          if (userId == null) {
            debugPrint('❌ Could not get user ID from response');
            return {
              'success': false,
              'error': 'Failed to get user ID',
            };
          }

          // Store tokens and user data securely
          await SecureStorageService.saveAuthData(
            accessToken: data['access'],
            refreshToken: data['refresh'],
            userId: userId,
            username: data['user']['username'] ?? '',
            email: data['user']['email'] ?? '',
          );

          debugPrint(
              '✅ Login successful and tokens stored with user ID: $userId');
          return {'success': true, 'data': data};
        }
      }

      debugPrint('❌ Login failed: ${response.statusCode}');
      return {
        'success': false,
        'error': response.data['detail'] ?? 'Authentication failed',
      };
    } on DioException catch (e) {
      debugPrint('❌ DioError during login: ${e.message}');
      if (e.type == DioExceptionType.connectionError) {
        return {
          'success': false,
          'error':
              'Unable to connect to server. Please check your internet connection.',
        };
      }
      // Check if response is HTML
      final contentType = e.response?.headers.value('content-type') ?? '';
      if (contentType.toLowerCase().contains('text/html')) {
        return {
          'success': false,
          'error': 'Server error occurred. Please try again later.',
        };
      }
      return {
        'success': false,
        'error': e.response?.data['detail'] ?? 'Network error occurred',
      };
    } catch (e) {
      debugPrint('❌ Unexpected error during login: $e');
      return {'success': false, 'error': 'An unexpected error occurred'};
    }
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
