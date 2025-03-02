import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../config.dart';

class AuthService {
  /// 🌐 HTTP client instance with default configuration
  static final Dio dio = Dio(BaseOptions(
    connectTimeout: Duration(seconds: 10),
    receiveTimeout: Duration(seconds: 10),
    validateStatus: (status) =>
        status != null && status < 500, // Accept all status codes below 500
  ));

  /// Stores the authentication token, username and email in shared preferences if successful
  static Future<bool> login(String username, String password) async {
    final url = '${Config.baseUrl}auth/login/';
    final Map<String, String> body = {
      'username': username,
      'password': password,
    };
    try {
      print('🔄 Attempting login with username: $username');
      final response = await dio.post(
        url,
        data: body,
      );
      print('📡 Server Response: ${response.statusCode} - ${response.data}');
      if (response.statusCode == 200 && response.data['token'] != null) {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('auth_token', response.data['token']);
        await prefs.setString('username', username);

        // Store email if it's included in the response
        if (response.data['email'] != null) {
          await prefs.setString('email', response.data['email']);
          print(
              '✅ Stored Email in SharedPreferences: ${response.data['email']}');
        }

        print('✅ Stored Token in SharedPreferences: ${response.data['token']}');
        print('✅ Stored Username in SharedPreferences: $username');
        return true;
      }
      print('⚠️ Login failed: ${response.data}');
      return false;
    } catch (e) {
      print('❌ Login error: ${e.toString()}');
      return false;
    }
  }

  /// 🔍 Retrieves the authentication token from shared preferences
  ///
  /// Returns null if no token is stored
  static Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('auth_token');
    print('📂 Retrieved Token from SharedPreferences: $token');
    return token;
  }

  /// 👤 Retrieves the username from shared preferences
  ///
  /// Returns null if no username is stored
  static Future<String?> getUsername() async {
    final prefs = await SharedPreferences.getInstance();
    final username = prefs.getString('username');
    print('📂 Retrieved Username from SharedPreferences: $username');
    return username;
  }

  /// 📧 Retrieves the email from shared preferences
  ///
  /// Returns null if no email is stored
  static Future<String?> getEmail() async {
    final prefs = await SharedPreferences.getInstance();
    final email = prefs.getString('email');
    print('📂 Retrieved Email from SharedPreferences: $email');
    return email;
  }

  /// 👨‍💼 Gets the current user profile information
  ///
  /// Returns a Map containing username, email and isLoggedIn status
  static Future<Map<String, dynamic>> getUserProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final username = prefs.getString('username');
    final email = prefs.getString('email');
    final token = prefs.getString('auth_token');

    return {
      'isLoggedIn': token != null,
      'username': username,
      'email': email,
    };
  }

  /// 🚪 Logs out the current user by removing all user data from shared preferences
  static Future<void> logout() async {
    print('🚪 Logging out user...');
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('auth_token');
    await prefs.remove('username');
    await prefs.remove('email');
    print('✅ User logged out. User data removed from SharedPreferences.');
  }

  /// 🏁 Checks if the user is currently logged in
  static Future<bool> isLoggedIn() async {
    final token = await getToken();
    bool loggedIn = token != null;
    print('🔍 Is user logged in? $loggedIn');
    return loggedIn;
  }

  /// 🛠️ Prints all stored keys and values in shared preferences for debugging
  static Future<void> printSharedPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    final keys = prefs.getKeys();
    print('🗄️ Shared Preferences Storage:');
    for (String key in keys) {
      print('🔑 $key: ${prefs.get(key)}');
    }
  }
}
