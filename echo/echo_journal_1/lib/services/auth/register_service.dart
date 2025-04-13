import 'package:dio/dio.dart';
import 'package:echo_journal1/features/authentication/pages/login_page.dart';
import 'package:flutter/material.dart';
import 'package:echo_journal1/core/configs/api_config.dart';

class RegisterService {
  static final _dio = Dio();

  static Future<Map<String, dynamic>> register(
    String username,
    String password,
    String email,
    BuildContext context,
  ) async {
    final url = ApiConfig.getFullUrl('auth/register/');

    final Map<String, dynamic> body = {
      'username': username,
      'password': password,
      'email': email,
    };

    try {
      print("Sending registration request to: $url");
      print("Request body: $body");

      Response response = await _dio.post(
        url,
        data: body,
        options: Options(
          headers: {'Content-Type': 'application/json'},
          validateStatus: (status) => true, // Accept all status codes
        ),
      );

      // Enhanced debugging logs
      print("Response Status Code: ${response.statusCode}");
      print("Response Data: ${response.data}");

      // Check for successful registration (201 status code and expected response format)
      if (response.statusCode == 201 &&
          response.data != null &&
          response.data['message'] != null &&
          response.data['email'] != null) {
        return {
          'success': true,
          'message': response.data['message'],
          'email': response.data['email']
        };
      } else {
        // Handle error cases
        String errorMessage = response.data['error'] ??
            response.data['message'] ??
            'Registration failed';
        print("Registration failed with error: $errorMessage");
        return {'success': false, 'error': errorMessage};
      }
    } on DioException catch (e) {
      print("DioError: ${e.toString()}");
      print("Error Type: ${e.type}");
      print("Error Message: ${e.message}");
      print("Error Response: ${e.response?.data}");

      if (e.type == DioExceptionType.connectionError) {
        return {
          'success': false,
          'error':
              'Unable to connect to server. Please check your internet connection.',
        };
      } else {
        String errorMessage = e.response?.data?['error'] ??
            e.response?.data?['message'] ??
            'Network error. Please try again.';
        return {'success': false, 'error': errorMessage};
      }
    } catch (e) {
      print("Unexpected Error: ${e.toString()}");
      return {'success': false, 'error': 'An unexpected error occurred'};
    }
  }

  static Future<bool> verifyOTP(String email, String otp) async {
    try {
      final url = ApiConfig.getFullUrl('auth/verify-otp/');
      final response = await _dio.post(
        url,
        data: {'email': email, 'otp': otp},
        options: Options(
          headers: {'Content-Type': 'application/json'},
          validateStatus: (status) => true,
        ),
      );

      return response.statusCode == 200;
    } on DioException catch (e) {
      print("DioError in verifyOTP: ${e.toString()}");
      return false;
    } catch (e) {
      print("Error in verifyOTP: ${e.toString()}");
      return false;
    }
  }

  static Future<Map<String, dynamic>> resendOTP(String email) async {
    try {
      final url = ApiConfig.getFullUrl('auth/resend-otp/');
      final response = await _dio.post(
        url,
        data: {'email': email},
        options: Options(
          headers: {'Content-Type': 'application/json'},
          validateStatus: (status) => true,
        ),
      );

      if (response.statusCode == 200) {
        return {'success': true};
      } else {
        return {
          'success': false,
          'error': response.data['message'] ?? 'Failed to resend OTP',
        };
      }
    } catch (e) {
      print("Error in resendOTP: ${e.toString()}");
      return {'success': false, 'error': 'Failed to resend OTP'};
    }
  }
}
