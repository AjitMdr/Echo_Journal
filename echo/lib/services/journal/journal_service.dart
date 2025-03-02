import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../config.dart';
import '../auth/login_service.dart';

/// 📓 Service class handling journal operations including fetching, creating,
/// updating, and deleting journals.
class JournalService {
  /// 🌐 Shared HTTP client instance with authentication interceptor
  static final Dio _dio = Dio(BaseOptions(
    baseUrl: Config.baseUrl,
    connectTimeout: Duration(seconds: 15),
    receiveTimeout: Duration(seconds: 15),
    validateStatus: (status) => status != null && status < 500,
  ))
    ..interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          final token = await AuthService.getToken();
          print('🔑 Stored Token: $token'); // Prints the token

          // Get preferred language
          final prefs = await SharedPreferences.getInstance();
          String? language = prefs.getString('language') ?? 'en';
          print(
              '🌍 Stored Language Preference: $language'); // Prints the stored language

          // Ensure token is added to request headers
          if (token != null && token.isNotEmpty) {
            options.headers['Authorization'] = 'Token $token';
          } else {
            print('⚠️ Token not found or invalid');
          }

          // Add language parameter to GET requests
          if (options.method == 'GET') {
            options.queryParameters = {
              ...options.queryParameters,
              'language': language,
            };
          }

          // Add language to request body for POST/PUT
          if ((options.method == 'POST' || options.method == 'PUT') &&
              options.data is Map) {
            options.data = {
              ...options.data,
              'language': language,
            };
          } else if ((options.method == 'POST' || options.method == 'PUT') &&
              options.data is String) {
            try {
              Map<String, dynamic> dataMap = jsonDecode(options.data);
              dataMap['language'] = language;
              options.data = jsonEncode(dataMap);
            } catch (e) {
              print('⚠️ Failed to add language to request body: $e');
            }
          }

          // Logging request details
          print(
              '📡 Sending journal request to: ${options.baseUrl}${options.path}');
          print('📝 Request Method: ${options.method}');
          print('📦 Request Data: ${options.data}');
          print('🔍 Query Parameters: ${options.queryParameters}');

          return handler.next(options);
        },
        onResponse: (response, handler) {
          // Logging response details
          print(
              '✅ Journal response from: ${response.requestOptions.baseUrl}${response.requestOptions.path}');
          print('📥 Response Status: ${response.statusCode}');
          print('📥 Response Data: ${response.data}');
          return handler.next(response);
        },
        onError: (DioException e, handler) {
          // Logging error details
          print(
              '❌ Journal request failed: ${e.requestOptions.baseUrl}${e.requestOptions.path}');
          print('🚨 Error: ${e.response?.data ?? e.message}');
          return handler.next(e);
        },
      ),
    );

  /// 📋 Fetches all journals from the server
  static Future<List<Map<String, dynamic>>> fetchJournals() async {
    try {
      print('🔄 Fetching journals...');
      final response = await _dio.get('journals/');

      if (response.statusCode == 200) {
        print('✅ Journals fetched successfully');
        return List<Map<String, dynamic>>.from(response.data);
      } else {
        throw Exception('Failed to fetch journals: ${response.statusCode}');
      }
    } catch (e) {
      print('❌ Journals fetch error: ${e.toString()}');
      return [];
    }
  }

  /// ✏️ Creates a new journal entry
  static Future<bool> createJournal(String title, String content) async {
    try {
      print('🔄 Creating journal with title: $title');
      final response = await _dio.post(
        'journals/',
        data: jsonEncode({
          'title': title,
          'content': content,
        }),
        options: Options(
          headers: {'Content-Type': 'application/json'},
        ),
      );

      if (response.statusCode == 201) {
        print('✅ Journal created successfully');
        return true;
      } else {
        throw Exception('Failed to create journal: ${response.data}');
      }
    } catch (e) {
      print('❌ Journal creation error: ${e.toString()}');
      if (e is DioException && e.response != null) {
        throw Exception(
            e.response?.data['error'] ?? 'Failed to create journal');
      }
      return false;
    }
  }

  /// 📝 Updates an existing journal entry
  static Future<bool> updateJournal(
      int journalId, String title, String content) async {
    try {
      print('🔄 Updating journal ID: $journalId with title: $title');
      final response = await _dio.put(
        'journals/$journalId/',
        data: jsonEncode({
          'title': title,
          'content': content,
        }),
        options: Options(
          headers: {'Content-Type': 'application/json'},
        ),
      );

      if (response.statusCode == 200) {
        print('✅ Journal updated successfully');
        return true;
      } else {
        throw Exception('Failed to update journal: ${response.data}');
      }
    } catch (e) {
      print('❌ Journal update error: ${e.toString()}');
      if (e is DioException && e.response != null) {
        throw Exception(
            e.response?.data['error'] ?? 'Failed to update journal');
      }
      return false;
    }
  }

  /// 🗑️ Deletes a journal entry
  static Future<bool> deleteJournal(int journalId) async {
    try {
      print('🔄 Deleting journal ID: $journalId');
      final response = await _dio.delete('journals/$journalId/');

      if (response.statusCode == 204) {
        print('✅ Journal deleted successfully');
        return true;
      } else {
        throw Exception('Failed to delete journal: ${response.statusCode}');
      }
    } catch (e) {
      print('❌ Journal deletion error: ${e.toString()}');
      if (e is DioException && e.response != null) {
        throw Exception(
            e.response?.data['error'] ?? 'Failed to delete journal');
      }
      return false;
    }
  }

  /// 🔍 Fetches a specific journal by ID
  /// 🔍 Fetches a specific journal by ID
  static Future<Map<String, dynamic>?> fetchJournalById(int journalId) async {
    try {
      print('🔄 Fetching journal ID: $journalId');
      final response = await _dio.get('journals/$journalId/');

      if (response.statusCode == 200) {
        print('✅ Journal fetched successfully');

        var data = response.data;
        return {
          'id': data['id'],
          'title': data['title'],
          'content': data['content'],
          'created_at': data['created_at'] ?? 'Unknown',
          'updated_at': data['updated_at'] ?? 'Unknown',
        };
      } else {
        throw Exception('Failed to fetch journal: ${response.statusCode}');
      }
    } catch (e) {
      print('❌ Journal fetch error: ${e.toString()}');
      return null;
    }
  }
}
