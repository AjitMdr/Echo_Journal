import 'package:dio/dio.dart';
import '../config.dart';

class AuthService {
  static final dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) => status != null && status < 500,
  ));

  // Login method you provided
  static Future<bool> login(String username, String password) async {
    final url = '${Config.baseUrl}/signup/initiate/';
    final Map<String, String> body = {
      'username': username,
      'password': password,
    };

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      if (response.statusCode == 200) {
        return true;
      } else {
        return false;
      }
    } catch (e) {
      print(e.toString());
      return false;
    }
  }

  // Verify OTP method
  static Future<bool> verifyOTP(String email, String otp) async {
    final url = '${Config.baseUrl}/signup/verify/';
    final Map<String, String> body = {
      'email': email,
      'otp': otp,
    };

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      if (response.statusCode == 201) {
        // Save token if provided in response
        if (response.data['token'] != null) {
          // You can implement token storage here
          // await SecureStorage.saveToken(response.data['token']);
        }
        return true;
      } else {
        print('Verification failed: ${response.data['error']}');
        return false;
      }
    } catch (e) {
      print('OTP verification error: ${e.toString()}');
      return false;
    }
  }

  // Resend OTP method
  static Future<bool> resendOTP(String email) async {
    final url = '${Config.baseUrl}/resend-otp/';
    final Map<String, String> body = {
      'email': email,
    };

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      if (response.statusCode == 200) {
        return true;
      } else {
        print('Resend OTP failed: ${response.data['error']}');
        return false;
      }
    } catch (e) {
      print('Resend OTP error: ${e.toString()}');
      return false;
    }
  }

  // Register method
  static Future<bool> register(String username, String password, String email) async {
    final url = '${Config.baseUrl}/register/';
    final Map<String, String> body = {
      'username': username,
      'password': password,
      'email': email,
    };

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      if (response.statusCode == 201) {
        return true;
      } else {
        print('Registration failed: ${response.data['error']}');
        return false;
      }
    } catch (e) {
      print('Registration error: ${e.toString()}');
      return false;
    }
  }
}