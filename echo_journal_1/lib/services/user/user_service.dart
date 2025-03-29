import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:echo_journal_1/services/auth/secure_storage_service.dart';
import 'package:echo_journal_1/core/configs/api_config.dart';
import '../../models/user.dart';

class UserService {
  final String authPrefix = '/auth';

  Future<String?> _getToken() async {
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

  Future<User> getCurrentUser() async {
    try {
      final token = await _getToken();
      final url = ApiConfig.getFullUrl('$authPrefix/users/profile/');

      final response = await http.get(
        Uri.parse(url),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        return User.fromJson(json.decode(response.body));
      } else if (response.statusCode == 401) {
        throw Exception('Authentication token expired or invalid');
      } else {
        throw Exception('Failed to load user profile: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString());
    }
  }

  Future<User> updateProfile({
    String? username,
    String? bio,
    String? profileImage,
  }) async {
    try {
      final token = await _getToken();
      final url = ApiConfig.getFullUrl('$authPrefix/profile/update/');

      final response = await http.put(
        Uri.parse(url),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
        body: json.encode({
          if (username != null) 'username': username,
          if (bio != null) 'bio': bio,
          if (profileImage != null) 'profile_image': profileImage,
        }),
      );

      if (response.statusCode == 200) {
        return User.fromJson(json.decode(response.body));
      } else {
        throw Exception('Failed to update profile: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception(e.toString());
    }
  }
}
