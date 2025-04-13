import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:echo_journal1/core/configs/api_config.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';

class TwoFactorService {
  static final String _authPrefix = '/auth';

  static Future<String?> _getToken() async {
    try {
      final token = await SecureStorageService.getAccessToken();
      if (token == null) {
        throw Exception('No authentication token found');
      }
      return token;
    } catch (e) {
      throw Exception('Failed to get authentication token: $e');
    }
  }

  static Future<bool> getTwoFactorStatus() async {
    try {
      final token = await _getToken();
      final url = ApiConfig.getFullUrl('$_authPrefix/2fa/status/');

      final response = await http.get(
        Uri.parse(url),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['is_enabled'] ?? false;
      } else {
        throw Exception('Failed to get 2FA status: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString());
    }
  }

  static Future<void> toggleTwoFactor(bool enable) async {
    try {
      final token = await _getToken();
      final url = ApiConfig.getFullUrl('$_authPrefix/2fa/toggle/');

      final response = await http.post(
        Uri.parse(url),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
        body: json.encode({
          'enable': enable,
        }),
      );

      if (response.statusCode != 200) {
        throw Exception('Failed to toggle 2FA: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString());
    }
  }
}
