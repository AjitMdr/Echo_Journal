import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:echo_journal1/core/configs/api_config.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';
import 'package:flutter/foundation.dart';
import 'package:dio/dio.dart';
import 'package:echo_journal1/services/auth/session_management.dart';
import 'package:echo_journal1/services/base_service.dart';

class ForgotPasswordService extends BaseService {
  static final String _authPrefix = 'auth';

  /// Request OTP for password reset
  static Future<Map<String, dynamic>> requestPasswordResetOTP(
      String email) async {
    try {
      final response = await BaseService.dio.post(
        '/auth/forgot_password/',
        data: {'email': email},
      );

      print('Request Password Reset OTP Response: ${response.data}');

      if (response.data['success'] == true) {
        return {
          'success': true,
          'message': response.data['message'] ?? 'OTP sent successfully',
          'error': null
        };
      } else {
        return {
          'success': false,
          'message': null,
          'error': response.data['error'] ?? 'Failed to send OTP'
        };
      }
    } on DioException catch (e) {
      print('Error in requestPasswordResetOTP: ${e.response?.data}');

      String errorMessage;
      if (e.type == DioExceptionType.connectionError ||
          e.error != null && e.error.toString().contains('SocketException')) {
        errorMessage = 'No internet connection. Please check your network.';
      } else if (e.response?.data != null &&
          e.response?.data['error'] != null) {
        errorMessage = e.response?.data['error'];
      } else {
        switch (e.type) {
          case DioExceptionType.connectionTimeout:
          case DioExceptionType.sendTimeout:
          case DioExceptionType.receiveTimeout:
            errorMessage =
                'Connection timeout. Please check your internet connection.';
            break;
          default:
            errorMessage = 'Failed to send OTP. Please try again.';
        }
      }
      return {'success': false, 'message': null, 'error': errorMessage};
    } catch (e) {
      print('Unexpected error in requestPasswordResetOTP: $e');
      return {
        'success': false,
        'message': null,
        'error': 'An unexpected error occurred. Please try again.'
      };
    }
  }

  /// Verify OTP and set new password
  static Future<Map<String, dynamic>> verifyOTPAndResetPassword({
    required String email,
    required String otp,
    required String newPassword,
  }) async {
    try {
      final response = await BaseService.dio.post(
        '/auth/verify-otp-reset-password/',
        data: {
          'email': email,
          'otp': otp,
          'new_password': newPassword,
        },
      );

      print('Verify OTP Response: ${response.data}');

      if (response.data['success'] == true) {
        // If tokens are provided in the response, save them
        if (response.data['refresh'] != null &&
            response.data['access'] != null) {
          await SecureStorageService.saveAuthData(
            accessToken: response.data['access'],
            refreshToken: response.data['refresh'],
            userId: response.data['user']['id'].toString(),
            username: response.data['user']['username'],
            email: response.data['user']['email'],
          );
        }

        return {
          'success': true,
          'message': response.data['message'] ?? 'Password reset successful',
          'error': null
        };
      } else {
        return {
          'success': false,
          'message': null,
          'error': response.data['error'] ?? 'Failed to reset password'
        };
      }
    } on DioException catch (e) {
      print('Error in verifyOTPAndResetPassword: ${e.response?.data}');
      String errorMessage;
      if (e.type == DioExceptionType.connectionError ||
          e.error != null && e.error.toString().contains('SocketException')) {
        errorMessage = 'No internet connection. Please check your network.';
      } else if (e.response?.data != null &&
          e.response?.data['error'] != null) {
        errorMessage = e.response?.data['error'];
      } else {
        errorMessage = 'Failed to reset password. Please try again.';
      }
      return {'success': false, 'message': null, 'error': errorMessage};
    } catch (e) {
      print('Unexpected error in verifyOTPAndResetPassword: $e');
      return {
        'success': false,
        'message': null,
        'error': 'An unexpected error occurred'
      };
    }
  }

  /// Resend OTP for password reset
  static Future<Map<String, dynamic>> resendPasswordResetOTP(
      String email) async {
    try {
      final response = await BaseService.dio.post(
        '/auth/resend-password-reset-otp/',
        data: {'email': email},
      );

      print('Resend OTP Response: ${response.data}');

      if (response.data['success'] == true) {
        return {
          'success': true,
          'message': response.data['message'] ?? 'OTP resent successfully',
          'error': null
        };
      } else {
        return {
          'success': false,
          'message': null,
          'error': response.data['error'] ?? 'Failed to resend OTP'
        };
      }
    } on DioException catch (e) {
      print('Error in resendPasswordResetOTP: ${e.response?.data}');
      String errorMessage;
      if (e.type == DioExceptionType.connectionError ||
          e.error != null && e.error.toString().contains('SocketException')) {
        errorMessage = 'No internet connection. Please check your network.';
      } else if (e.response?.data != null &&
          e.response?.data['error'] != null) {
        errorMessage = e.response?.data['error'];
      } else {
        errorMessage = 'Failed to resend OTP';
      }
      return {'success': false, 'message': null, 'error': errorMessage};
    } catch (e) {
      print('Unexpected error in resendPasswordResetOTP: $e');
      return {
        'success': false,
        'message': null,
        'error': 'An unexpected error occurred'
      };
    }
  }
}
