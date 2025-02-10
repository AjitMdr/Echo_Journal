import 'package:dio/dio.dart';
import '../config.dart';

class AuthService {
  static final dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) => status != null && status < 500, // Accept all status codes below 500
  ));

  static Future<bool> login(String username, String password) async {
    final url = '${Config.baseUrl}/login/';
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
        return true; // Return true when login is successful
      } else {
        return false; // Return false when login fails
      }
    } catch (e) {
      print(e.toString());
      return false; // Return false in case of an error
    }
  }
}
