import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_fe/services/auth/secure_storage_service.dart';
import 'package:flutter/foundation.dart';
import 'package:echo_fe/services/auth/session_management.dart';
import 'package:echo_fe/core/configs/api_config.dart';

class AuthService {
  /// 🌐 HTTP client instance with default configuration
  static final Dio dio = Dio(
    BaseOptions(
      connectTimeout: Duration(seconds: 10),
      receiveTimeout: Duration(seconds: 10),
      validateStatus: (status) => status != null && status < 500,
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
          validateStatus: (status) => status != null && status < 500,
          followRedirects: false,
          headers: {'Accept': 'application/json'},
        ),
      );

      debugPrint('📡 Login Response Status: ${response.statusCode}');
      debugPrint('📡 Login Response Data: ${response.data}');

      // Check if response is HTML (indicates server error)
      final contentType = response.headers.value('content-type') ?? '';
      if (contentType.toLowerCase().contains('text/html')) {
        debugPrint('❌ Received HTML response instead of JSON');
        return {
          'success': false,
          'error': 'Server error occurred. Please try again later.',
        };
      }

      if (response.statusCode == 200) {
        final data = response.data;

        // First check if we have user data and tokens
        if (data['user'] == null ||
            data['access'] == null ||
            data['refresh'] == null) {
          debugPrint('❌ Missing required data in response');
          return {
            'success': false,
            'error': 'Invalid response from server',
          };
        }

        // Then check if account is verified
        if (data['user']['is_verified'] == false) {
          debugPrint('❌ Account not verified');
          return {
            'success': false,
            'error':
                'Please verify your account first. Check your email for the verification OTP.',
            'needs_verification': true,
            'email': data['user']['email']
          };
        }

        // Get user ID
        String? userId;
        if (data['user']['id'] != null) {
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

        // Only store tokens and user data if account is verified
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

      // Handle error response
      debugPrint('❌ Login failed: ${response.statusCode}');
      if (response.data is Map && response.data['error'] is Map) {
        final errorMap = response.data['error'] as Map;
        if (errorMap['non_field_errors'] is List) {
          final errorMessage = errorMap['non_field_errors'][0];
          // Check if it's a verification error
          if (errorMessage
              .toString()
              .toLowerCase()
              .contains('verify your account')) {
            return {
              'success': false,
              'error': errorMessage,
              'needs_verification': true,
              'email': username
            };
          }
          return {'success': false, 'error': errorMessage};
        }
      }
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
      // Handle error response structure
      if (e.response?.data is Map && e.response?.data['error'] is Map) {
        final errorMap = e.response?.data['error'] as Map;
        if (errorMap['non_field_errors'] is List) {
          final errorMessage = errorMap['non_field_errors'][0];
          // Check if it's a verification error
          if (errorMessage
              .toString()
              .toLowerCase()
              .contains('verify your account')) {
            return {
              'success': false,
              'error': errorMessage,
              'needs_verification': true,
              'email': username
            };
          }
          return {'success': false, 'error': errorMessage};
        }
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
