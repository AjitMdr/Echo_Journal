import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../config.dart';
import '../auth/login_service.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';

/// 👤 Service class handling user profile operations including fetching, updating,
/// and password management.
class ProfileService {
  /// 🌐 Shared HTTP client instance with authentication interceptor
  static final Dio _dio = Dio(BaseOptions(
    baseUrl: "${Config.baseUrl}auth/",
    connectTimeout: Duration(seconds: 15),
    receiveTimeout: Duration(seconds: 15),
    validateStatus: (status) => status != null && status < 500,
  ))
    ..interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          final token = await AuthService.getToken();
          if (token != null) {
            options.headers['Authorization'] = 'Token $token';
          }

          // Logging request details
          print('📡 Sending request to: ${options.baseUrl}${options.path}');
          print('📝 Request Method: ${options.method}');
          print('📦 Request Data: ${options.data}');

          return handler.next(options);
        },
        onResponse: (response, handler) {
          // Logging response details
          print(
              '✅ Response from: ${response.requestOptions.baseUrl}${response.requestOptions.path}');
          print('📥 Response Data: ${response.data}');
          return handler.next(response);
        },
        onError: (DioException e, handler) {
          // Logging error details
          print(
              '❌ Request failed: ${e.requestOptions.baseUrl}${e.requestOptions.path}');
          print('🚨 Error: ${e.response?.data ?? e.message}');
          return handler.next(e);
        },
      ),
    );

  /// 📋 Fetches the user profile information from the server
  static Future<Map<String, dynamic>> fetchProfile() async {
    try {
      print('🔄 Fetching user profile...');
      final response = await _dio.get('profile/');

      if (response.statusCode == 200) {
        print('✅ Profile fetched successfully');

        final prefs = await SharedPreferences.getInstance();
        if (response.data['username'] != null) {
          await prefs.setString('username', response.data['username']);
        }
        if (response.data['email'] != null) {
          await prefs.setString('email', response.data['email']);
        }
        if (response.data['profile_picture'] != null) {
          await prefs.setString(
              'profile_picture', response.data['profile_picture']);
          print(
              '🖼️ Profile Picture URL fetched: ${response.data['profile_picture']}');
        }

        await prefs.setString('profile_data', jsonEncode(response.data));
        print('💾 Profile data saved to SharedPreferences');
        return response.data;
      } else {
        throw Exception('Failed to fetch profile: ${response.statusCode}');
      }
    } catch (e) {
      print('❌ Profile fetch error: ${e.toString()}');
      throw Exception('Failed to fetch profile: ${e.toString()}');
    }
  }

  /// 📝 Updates user profile information
  static Future<Map<String, dynamic>> updateProfile(
      Map<String, dynamic> profileData) async {
    try {
      print('🔄 Updating profile with data: $profileData');
      final response = await _dio.put('profile/update/', data: profileData);

      if (response.statusCode == 200) {
        print('✅ Profile updated successfully');

        // Cache the updated profile data
        await _cacheProfileData(response.data);

        if (profileData.containsKey('username') &&
            profileData['username'] != null) {
          final prefs = await SharedPreferences.getInstance();
          await prefs.setString('username', profileData['username']);
          print(
              '✅ Username updated in SharedPreferences: ${profileData['username']}');
        }

        return response.data;
      } else {
        throw Exception('Failed to update profile: ${response.data}');
      }
    } catch (e) {
      print('❌ Profile update error: ${e.toString()}');
      throw Exception('Failed to update profile: ${e.toString()}');
    }
  }

  /// 🔑 Changes the user password
  static Future<String> changePassword(
      String currentPassword, String newPassword) async {
    try {
      print('🔄 Attempting to change password...');
      final response = await _dio.put(
        'profile/password/',
        data: {
          'current_password': currentPassword,
          'new_password': newPassword,
        },
      );

      if (response.statusCode == 200) {
        print('✅ Password changed successfully');
        return response.data['message'] ?? 'Password updated successfully';
      } else {
        throw Exception(response.data['error'] ?? 'Failed to change password');
      }
    } catch (e) {
      print('❌ Password change error: ${e.toString()}');
      if (e is DioException && e.response != null) {
        throw Exception(
            e.response?.data['error'] ?? 'Failed to change password');
      }
      throw Exception('Failed to change password: ${e.toString()}');
    }
  }

  static Future<Map<String, dynamic>> updateProfilePicture(
      File imageFile) async {
    try {
      print('🔄 Starting profile picture update process...');
      print('📂 Original image path: ${imageFile.path}');

      // Get stored auth token
      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString('auth_token');
      if (token == null) throw Exception('Authentication token not found');
      print('🔑 Retrieved auth token: ${token.substring(0, 10)}...');

      // Compress the image
      String dir = imageFile.path
          .split('/')
          .sublist(0, imageFile.path.split('/').length - 1)
          .join('/');
      String fileName = imageFile.path.split('/').last;
      String fileExtension = fileName.split('.').last.toLowerCase();
      print('📁 Directory: $dir');
      print('📄 Filename: $fileName');
      print('📎 File extension: $fileExtension');

      // Create compressed file
      File compressedFile = await compressImage(imageFile, dir, fileExtension);
      print('🗜️ Compressed file path: ${compressedFile.path}');
      print('📊 Original size: ${await imageFile.length()} bytes');
      print('📊 Compressed size: ${await compressedFile.length()} bytes');

      // Create form data
      FormData formData = FormData.fromMap({
        'profile_picture': await MultipartFile.fromFile(
          compressedFile.path,
          filename: 'profile_$fileExtension',
        ),
      });
      print('📦 Form data created with filename: profile_$fileExtension');

      String endpoint = 'profile/picture/';
      print('🌐 Full request URL: ${_dio.options.baseUrl}$endpoint');

      final response = await _dio.post(
        endpoint,
        data: formData,
        options: Options(
          headers: {
            'Authorization': 'Token $token',
          },
        ),
      );

      print('📥 Raw response data: ${response.data}');
      print('🔍 Response status code: ${response.statusCode}');

      if (response.statusCode == 200) {
        print('✅ Profile picture upload successful');

        // Extract profile picture URL
        String? pictureUrl;
        if (response.data['data'] != null &&
            response.data['data']['profile_picture'] != null) {
          pictureUrl = response.data['data']['profile_picture'];
        } else if (response.data['profile_picture'] != null) {
          pictureUrl = response.data['profile_picture'];
        }

        print('🖼️ Extracted picture URL: $pictureUrl');

        // Cache the updated profile data
        Map<String, dynamic> dataToCache;
        if (response.data['data'] != null) {
          dataToCache = response.data['data'];
        } else {
          dataToCache = response.data;
        }

        print('💾 Caching profile data: $dataToCache');
        await _cacheProfileData(dataToCache);

        return dataToCache;
      } else {
        print('❌ Upload failed with status code: ${response.statusCode}');
        print('❌ Error response: ${response.data}');
        throw Exception(
            'Failed to update profile picture: ${response.data['error']}');
      }
    } catch (e) {
      print('❌ Profile picture update error: $e');
      throw Exception('Failed to update profile picture: $e');
    }
  }

  /// 🗜️ Compresses an image file
  static Future<File> compressImage(
      File file, String dir, String extension) async {
    // Ensure file has a valid .jpg or .jpeg extension
    if (extension != 'jpg' && extension != 'jpeg') {
      extension = 'jpeg'; // Default to jpeg if not jpg/jpeg
    }

    final compressedFile = await FlutterImageCompress.compressAndGetFile(
      file.path,
      '$dir/compressed_image.$extension',
      quality: 70,
      minWidth: 800,
      minHeight: 800,
    );

    if (compressedFile == null) {
      // If compression fails, return original
      return file;
    }

    return File(compressedFile.path);
  }

  /// 💾 Caches profile data
  static Future<void> _cacheProfileData(
      Map<String, dynamic> profileData) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('profile_data', jsonEncode(profileData));
    print('💾 Profile data cached to SharedPreferences');
  }

  /// 👤 Gets the cached profile data
  static Future<Map<String, dynamic>?> getCachedProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final profileDataString = prefs.getString('profile_data');

    if (profileDataString != null) {
      try {
        return jsonDecode(profileDataString) as Map<String, dynamic>;
      } catch (e) {
        print('⚠️ Error parsing cached profile data: ${e.toString()}');
        return null;
      }
    }
    return null;
  }

  /// 🖼️ Gets the profile picture URL from cache
  static Future<String?> getProfilePictureUrl() async {
    final profileData = await getCachedProfile();
    return profileData?['profile_picture'];
  }
}
