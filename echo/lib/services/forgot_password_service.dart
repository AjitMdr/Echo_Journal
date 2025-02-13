import 'package:dio/dio.dart';
import '../config.dart';

class AuthService {
  static final dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) =>
        status != null && status < 500, // Accept all status codes below 500
  ));

  // Forgot password service - initiates OTP
  static Future<Map<String, dynamic>> forgotPassword(String email) async {
    final url = '${Config.baseUrl}/forgot_password/';
    final Map<String, String> body = {
      'email': email,
    };

    // Printing the email being sent
    print("Sending request with email: $email");

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      // Debugging logs: printing response status code and data
      print("Response Status Code: ${response.statusCode}");
      print("Response Data: ${response.data}");

      return {
        'success': response.statusCode == 200,
        'data': response.data,
      };
    } catch (e) {
      print("Error: ${e.toString()}");
      return {
        'success': false,
        'error': 'Network error. Please try again.',
      };
    }
  }

  // Reset password service with OTP verification
  static Future<Map<String, dynamic>> resetPassword(
    String email,
    String otp,
    String newPassword,
  ) async {
    final url = '${Config.baseUrl}/verify-otp-reset-password/';
    final Map<String, dynamic> body = {
      'email': email,
      'otp': otp,
      'new_password': newPassword,
    };

    // Printing the data being sent
    print("Sending request to reset password with email: $email, OTP: $otp");

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      // Debugging logs: printing response status code and data
      print("Response Status Code: ${response.statusCode}");
      print("Response Data: ${response.data}");

      return {
        'success': response.statusCode == 200,
        'data': response.data,
      };
    } catch (e) {
      print("Error: ${e.toString()}");
      return {
        'success': false,
        'error': 'Network error. Please try again.',
      };
    }
  }

  // Optional: Add resend OTP service
  static Future<Map<String, dynamic>> resendOTP(String email) async {
    final url = '${Config.baseUrl}/resend-password-reset-otp/';
    final Map<String, String> body = {
      'email': email,
    };

    print("Requesting new OTP for email: $email");

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      print("Response Status Code: ${response.statusCode}");
      print("Response Data: ${response.data}");

      return {
        'success': response.statusCode == 200,
        'data': response.data,
      };
    } catch (e) {
      print("Error: ${e.toString()}");
      return {
        'success': false,
        'error': 'Network error. Please try again.',
      };
    }
  }
}
