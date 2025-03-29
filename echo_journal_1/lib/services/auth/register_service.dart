import 'package:dio/dio.dart';
import 'package:echo_journal_1/features/authentication/pages/login_page.dart';
import 'package:flutter/material.dart';
import 'package:echo_journal_1/core/configs/api_config.dart';

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
      Response response = await _dio.post(
        url,
        data: body,
        options: Options(
          headers: {'Content-Type': 'application/json'},
          validateStatus: (status) => true, // Accept all status codes
        ),
      );

      // Debugging logs
      print("Response Status Code: ${response.statusCode}");
      print("Response Data: ${response.data}");

      if (response.statusCode == 201) {
        // Show success message
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Registration successful! Please login.'),
            backgroundColor: Colors.green,
          ),
        );

        // Navigate to login page
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const LoginPage()),
        );

        return {'success': true, 'data': response.data};
      } else {
        return {
          'success': false,
          'error': response.data['message'] ?? 'Server error occurred',
        };
      }
    } on DioException catch (e) {
      print("DioError: ${e.toString()}");
      if (e.type == DioExceptionType.connectionError) {
        return {
          'success': false,
          'error':
              'Unable to connect to server. Please check your internet connection.',
        };
      } else {
        print("Error Response: ${e.response?.data}");
        return {'success': false, 'error': 'Network error. Please try again.'};
      }
    } catch (e) {
      print("Error: ${e.toString()}");
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
