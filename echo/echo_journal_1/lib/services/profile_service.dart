import 'dart:io';
import 'package:dio/dio.dart';
import 'package:http_parser/http_parser.dart';
import 'package:echo_journal1/core/configs/api_config.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';
import '../constants/api_constants.dart';
import 'package:flutter/foundation.dart';
import 'dart:math' as math;

class ProfileService {
  static final Dio _dio = Dio();

  static Future<void> _debugPrintToken() async {
    await SecureStorageService.debugPrintAllValues();
  }

  static Future<String?> _getToken() async {
    final token = await SecureStorageService.getAccessToken();
    debugPrint(
        '🔑 Token retrieved: ${token != null ? "Token exists" : "Token is null"}');
    if (token != null) {
      debugPrint(
          '🔑 Token starts with: ${token.substring(0, math.min(10, token.length))}...');
    }
    return token;
  }

  static Future<String?> _refreshToken() async {
    try {
      final refreshToken = await SecureStorageService.getRefreshToken();
      if (refreshToken == null) {
        debugPrint('❌ No refresh token found');
        return null;
      }

      debugPrint(
          '🔄 Attempting to refresh token with refresh token: ${refreshToken.substring(0, math.min(10, refreshToken.length))}...');

      final response = await _dio.post(
        '${ApiConfig.baseUrl}/api/auth/token/refresh/',
        data: {'refresh': refreshToken},
      );

      debugPrint('🔄 Refresh token response: ${response.data}');

      if (response.statusCode == 200) {
        final newAccessToken = response.data['access'];
        await SecureStorageService.setAccessToken(newAccessToken);
        debugPrint('✅ Token refreshed successfully');
        return newAccessToken;
      }
    } catch (e) {
      debugPrint('❌ Error refreshing token: $e');
    }
    return null;
  }

  static Future<Map<String, dynamic>> updateProfilePicture(
      File imageFile) async {
    try {
      debugPrint('🔍 Starting profile picture update...');
      await _debugPrintToken();

      final token = await _getToken();
      if (token == null) {
        debugPrint('❌ No token found in secure storage');
        throw DioException(
          requestOptions: RequestOptions(path: ''),
          error: 'No authentication token found',
          type: DioExceptionType.unknown,
        );
      }

      String fileName = imageFile.path.split('/').last;
      String fileExtension = fileName.split('.').last.toLowerCase();
      debugPrint('📁 Uploading file: $fileName with extension: $fileExtension');

      String mimeType = 'jpeg'; // default
      if (fileExtension == 'png') {
        mimeType = 'png';
      } else if (fileExtension == 'gif') {
        mimeType = 'gif';
      }

      FormData formData = FormData.fromMap({
        'profile_picture': await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
          contentType: MediaType('image', mimeType),
        ),
      });

      final url = '${ApiConfig.baseUrl}/api/auth/profile/picture/';
      debugPrint('🌐 Making request to: $url');
      debugPrint(
          '🔐 Using Authorization header: Bearer ${token.substring(0, math.min(10, token.length))}...');

      Response? response;
      try {
        response = await _dio.post(
          url,
          data: formData,
          options: Options(
            headers: {
              'Authorization': 'Bearer $token',
              'Accept': 'application/json',
            },
            validateStatus: (status) => true,
          ),
        );
      } catch (e) {
        debugPrint('❌ Request error: $e');
        throw e;
      }

      debugPrint('📡 Response status code: ${response.statusCode}');
      debugPrint('📡 Response headers: ${response.headers}');
      debugPrint('📡 Response data: ${response.data}');

      if (response.statusCode == 200 || response.statusCode == 201) {
        return response.data;
      } else if (response.statusCode == 401) {
        debugPrint('❌ Authentication failed. Response data: ${response.data}');
        // Try to refresh the token
        final newToken = await _refreshToken();
        if (newToken != null) {
          debugPrint('🔄 Token refreshed, retrying request with new token');
          // Retry the request with the new token
          final retryResponse = await _dio.post(
            url,
            data: formData,
            options: Options(
              headers: {
                'Authorization': 'Bearer $newToken',
                'Accept': 'application/json',
              },
            ),
          );

          debugPrint(
              '📡 Retry response status code: ${retryResponse.statusCode}');
          debugPrint('📡 Retry response data: ${retryResponse.data}');

          if (retryResponse.statusCode == 200 ||
              retryResponse.statusCode == 201) {
            return retryResponse.data;
          }
        }
        throw DioException(
          requestOptions: RequestOptions(path: ''),
          error: 'Session expired. Please log in again.',
          type: DioExceptionType.unknown,
        );
      } else {
        throw DioException(
          requestOptions: RequestOptions(path: ''),
          error:
              'Failed to update profile picture. Status: ${response.statusCode}',
          type: DioExceptionType.unknown,
        );
      }
    } on DioException catch (e) {
      debugPrint('❌ DioException: ${e.message}');
      debugPrint('❌ DioException response: ${e.response?.data}');
      if (e.response?.statusCode == 401) {
        throw 'Session expired. Please log in again.';
      }
      throw 'Failed to update profile picture: ${e.message}';
    } catch (e) {
      debugPrint('❌ General error: $e');
      throw 'Failed to update profile picture: $e';
    }
  }

  static Future<Map<String, dynamic>> getProfile() async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('No token found');
      }

      final response = await _dio.get(
        '${ApiConfig.baseUrl}/api/users/profile/',
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'Accept': 'application/json',
          },
        ),
      );

      if (response.statusCode == 401) {
        throw Exception('Unauthorized: Please log in again');
      }

      if (response.statusCode != 200) {
        throw Exception('Failed to get profile');
      }

      return response.data;
    } catch (e) {
      throw Exception('Error getting profile: $e');
    }
  }

  static Future<Map<String, dynamic>> updateProfile(
      Map<String, dynamic> data) async {
    try {
      final token = await _getToken();
      if (token == null) {
        throw Exception('No token found');
      }

      final response = await _dio.patch(
        '${ApiConfig.baseUrl}/api/users/profile/update/',
        data: data,
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'Accept': 'application/json',
          },
        ),
      );

      if (response.statusCode == 401) {
        throw Exception('Unauthorized: Please log in again');
      }

      if (response.statusCode != 200) {
        throw Exception('Failed to update profile');
      }

      return response.data;
    } catch (e) {
      throw Exception('Error updating profile: $e');
    }
  }
}
