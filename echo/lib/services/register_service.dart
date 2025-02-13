import 'package:dio/dio.dart';
import '../config.dart';

class AuthService {
  static final dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) => status != null && status < 500, // Accept all status codes below 500
  ));

  static Future<Map<String, dynamic>> register(String username, String password, String email) async {
    final url = '${Config.baseUrl}/signup/initiate/';
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

      // Debugging logs
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
