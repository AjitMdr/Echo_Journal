import 'package:dio/dio.dart';
import 'package:echo_fe/core/configs/api_config.dart';

class OTPService {
  static final Dio dio = Dio(
    BaseOptions(
      connectTimeout: Duration(seconds: 10),
      receiveTimeout: Duration(seconds: 10),
      validateStatus: (status) => status != null && status < 500,
    ),
  );

  // Verify OTP
  static Future<bool> verifyOTP(String email, String otp) async {
    try {
      print('🔄 Verifying OTP for email: $email'); // Debug log

      final response = await dio.post(
        ApiConfig.getFullUrl('auth/signup/verify/'),
        data: {
          'email': email,
          'otp': otp,
        },
        options: Options(
          headers: {'Content-Type': 'application/json'},
          validateStatus: (status) => true,
        ),
      );

      print('📡 Verify OTP Response: ${response.statusCode}'); // Debug log
      print('📡 Verify OTP Data: ${response.data}'); // Debug log

      return response.statusCode == 200;
    } on DioException catch (e) {
      print('❌ Error verifying OTP: ${e.message}'); // Debug log
      if (e.response?.data != null) {
        print('❌ Error response: ${e.response?.data}'); // Debug log
        throw Exception(
          e.response?.data['error'] ?? 'Network error occurred',
        );
      }
      throw Exception('Network error occurred');
    }
  }

  // Resend OTP
  static Future<Map<String, dynamic>> resendOTP(String email) async {
    try {
      print('🔄 Resending OTP for email: $email'); // Debug log

      final response = await dio.post(
        ApiConfig.getFullUrl('auth/resend_otp/'),
        data: {'email': email},
      );

      print('📡 Resend OTP Response: ${response.statusCode}'); // Debug log
      print('📡 Resend OTP Data: ${response.data}'); // Debug log

      if (response.statusCode == 200) {
        return {'success': true, 'message': 'OTP resent successfully'};
      }

      return {
        'success': false,
        'error': response.data['error'] ?? 'Failed to resend OTP'
      };
    } on DioException catch (e) {
      print('❌ Error resending OTP: ${e.message}'); // Debug log
      if (e.response?.data != null) {
        print('❌ Error response: ${e.response?.data}'); // Debug log
        return {
          'success': false,
          'error': e.response?.data['error'] ?? 'Network error occurred'
        };
      }
      return {'success': false, 'error': 'Network error occurred'};
    }
  }
}
