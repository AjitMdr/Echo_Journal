import 'package:dio/dio.dart';
import '../config.dart';

class AuthService {
  static final dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) =>
        status != null && status < 500, // Accept all status codes below 500
  ));

  static Future<bool> login(String username, String password) async {
    final url = '${Config.baseUrl}/login/';
    final Map<String, String> body = {
      'username': username,
      'password': password,
    };

    // Print what is being sent in the request
    print('Sending request to $url');
    print('Request body: $body');

    try {
      Response response = await dio.post(
        url,
        data: body,
      );

      // Print the response status and data
      print('Response status: ${response.statusCode}');
      print('Response data: ${response.data}');

      if (response.statusCode == 200) {
        return true;
      } else {
        return false;
      }
    } catch (e) {
      print('Error: ${e.toString()}');
      return false;
    }
  }
}
