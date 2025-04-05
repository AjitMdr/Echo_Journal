import 'package:dio/dio.dart';
import 'package:echo_fe/core/configs/api_config.dart';
import 'package:flutter/foundation.dart';

class VerifyAccountService {
  static final _dio = Dio();

  static Future<Map<String, dynamic>> verifyAccount(
    String username,
    String password,
  ) async {
    final url = ApiConfig.getFullUrl('/auth/verify-account/');

    try {
      debugPrint('🔐 Attempting to verify account for user: $username');
      debugPrint('🌐 Verify URL: $url');

      final response = await _dio.post(
        url,
        data: {
          'username': username,
          'password': password,
        },
        options: Options(
          headers: {'Content-Type': 'application/json'},
          validateStatus: (status) => true,
        ),
      );

      debugPrint('📡 Verify Response Status: ${response.statusCode}');
      debugPrint('📡 Verify Response Data: ${response.data}');

      if (response.statusCode == 200) {
        return {
          'success': true,
          'email': response.data['email'],
          'message': response.data['message']
        };
      } else {
        String errorMessage = response.data['error'] ??
            response.data['detail'] ??
            'Verification failed';

        if (response.data is Map && response.data['error'] is Map) {
          final errorMap = response.data['error'] as Map;
          if (errorMap['non_field_errors'] is List) {
            errorMessage = errorMap['non_field_errors'][0];
          }
        }

        return {'success': false, 'error': errorMessage};
      }
    } on DioException catch (e) {
      debugPrint('❌ DioError during verification: ${e.message}');
      if (e.type == DioExceptionType.connectionError) {
        return {
          'success': false,
          'error':
              'Unable to connect to server. Please check your internet connection.',
        };
      }
      return {
        'success': false,
        'error': e.response?.data?['detail'] ?? 'Network error occurred',
      };
    } catch (e) {
      debugPrint('❌ Unexpected error during verification: $e');
      return {'success': false, 'error': 'An unexpected error occurred'};
    }
  }
}
