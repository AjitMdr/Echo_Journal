import 'package:dio/dio.dart';
import '../config.dart';

class AuthService {
  static final dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) => status != null && status < 500,
  ));

  static void _debugRequest(
      String method, String url, Map<String, String> body) {
    print('[$method] Request to: $url');
    print('Body: $body');
  }

  // Login method
  static Future<bool> login(String username, String password) async {
    final url = '${Config.baseUrl}/signup/initiate/';
    final Map<String, String> body = {
      'username': username,
      'password': password,
    };

    _debugRequest('POST', url, body);

    try {
      Response response = await dio.post(url, data: body);

      print('Response: ${response.statusCode} - ${response.data}');

      return response.statusCode == 200;
    } catch (e) {
      print('Login error: ${e.toString()}');
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

    _debugRequest('POST', url, body);

    try {
      Response response = await dio.post(url, data: body);

      print('Response: ${response.statusCode} - ${response.data}');

      if (response.statusCode == 201 && response.data['token'] != null) {
        // You can implement token storage here
        // await SecureStorage.saveToken(response.data['token']);
        return true;
      }
      return false;
    } catch (e) {
      print('OTP verification error: ${e.toString()}');
      return false;
    }
  }

  // Resend OTP method
  static Future<bool> resendOTP(String email) async {
    final url = '${Config.baseUrl}/resend_otp/';
    final Map<String, String> body = {
      'email': email,
    };

    _debugRequest('POST', url, body);

    try {
      Response response = await dio.post(url, data: body);

      print('Response: ${response.statusCode} - ${response.data}');

      return response.statusCode == 200;
    } catch (e) {
      print('Resend OTP error: ${e.toString()}');
      return false;
    }
  }

  // Register method
  static Future<bool> register(
      String username, String password, String email) async {
    final url = '${Config.baseUrl}/register/';
    final Map<String, String> body = {
      'username': username,
      'password': password,
      'email': email,
    };

    _debugRequest('POST', url, body);

    try {
      Response response = await dio.post(url, data: body);

      print('Response: ${response.statusCode} - ${response.data}');

      return response.statusCode == 201;
    } catch (e) {
      print('Registration error: ${e.toString()}');
      return false;
    }
  }
}
