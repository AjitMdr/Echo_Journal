import 'package:dio/dio.dart';
import '../config.dart';

class AuthService {
  static final dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) => status != null && status < 500, // Accept all status codes below 500
  ));

  static Future<bool> register(String username, String password, String email) async {
    final url = '${Config.baseUrl}/signup';
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

      // Print the response details for debugging
      print("Response Status Code: ${response.statusCode}");
      
      print("Response Headers: ${response.headers}");

      if (response.statusCode == 201) {
        return true; // Return true when registration is successful
      } else {
        return false; // Return false when registration fails
      }
    } catch (e) {
      print("Error: ${e.toString()}");
      return false; // Return false in case of an error
    }
  }
}
