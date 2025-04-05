import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:echo_fe/services/auth/secure_storage_service.dart';
import 'package:echo_fe/core/configs/api_config.dart';

class JournalService {
  final Dio _dio = Dio();

  JournalService() {
    _setupDio();
  }

  void _setupDio() {
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          // Add auth token to every request
          final token = await SecureStorageService.getAccessToken();
          if (token != null) {
            options.headers['Authorization'] = 'Bearer $token';
            debugPrint(
                '🔒 Added auth token to journal request: ${options.uri}');
          } else {
            debugPrint(
                '❌ No auth token available for journal request: ${options.uri}');
          }
          return handler.next(options);
        },
        onError: (DioException e, handler) {
          debugPrint('🚨 Journal DioError: ${e.type} - ${e.message}');
          debugPrint('🚨 Request: ${e.requestOptions.uri}');
          if (e.response != null) {
            debugPrint(
                '🚨 Response: ${e.response?.statusCode} - ${e.response?.data}');
          }
          return handler.next(e);
        },
      ),
    );
  }

  Future<Map<String, dynamic>> getJournals({
    int page = 1,
    String? search,
  }) async {
    try {
      final url = ApiConfig.getFullUrl(ApiConfig.journalEndpoint);
      debugPrint('Fetching journals from: $url');

      final response = await _dio.get(
        url,
        options: Options(
          validateStatus: (status) {
            debugPrint('Response status: $status');
            return status! < 500;
          },
        ),
        queryParameters: {
          'page': page,
          if (search != null && search.isNotEmpty) 'search': search,
        },
      );

      debugPrint('Response data: ${response.data}');

      if (response.statusCode == 401) {
        throw Exception('Authentication failed. Please log in again.');
      }

      if (response.statusCode != 200) {
        throw Exception(
            'Failed to load journals: ${response.statusCode} - ${response.data}');
      }

      return response.data;
    } catch (e) {
      debugPrint('Error in getJournals: $e');
      throw Exception('Failed to load journals: $e');
    }
  }

  Future<Map<String, dynamic>> getJournal(int id) async {
    try {
      final url = ApiConfig.getFullUrl('journal/$id/');
      print('Fetching journal details from: $url');

      final response = await _dio.get(
        url,
        options: Options(
          validateStatus: (status) {
            print('Response status: $status');
            return status! < 500;
          },
        ),
      );

      if (response.statusCode != 200) {
        throw Exception(
            'Failed to load journal: ${response.statusCode} - ${response.data}');
      }

      print('Journal details response: ${response.data}');
      return response.data;
    } catch (e) {
      print('Error in getJournal: $e');
      throw Exception('Failed to load journal: $e');
    }
  }

  Future<Map<String, dynamic>> createJournal(Map<String, dynamic> data) async {
    try {
      // Validate required fields
      if (data['title']?.trim().isEmpty ?? true) {
        throw Exception('Title is required');
      }
      if (data['content']?.trim().isEmpty ?? true) {
        throw Exception('Content is required');
      }
      if (!['en', 'ne'].contains(data['language'])) {
        throw Exception('Invalid language. Must be either "en" or "ne"');
      }

      final response = await _dio.post(
        ApiConfig.getFullUrl('journal/'),
        data: {
          'title': data['title'].trim(),
          'content': data['content'].trim(),
          'language': data['language'],
        },
        options: Options(
          validateStatus: (status) => status == 201 || status == 400,
        ),
      );

      if (response.statusCode == 400) {
        final errors = response.data['errors'] ?? response.data;
        final message = response.data['message'] ??
            (errors is Map
                ? errors.values.join(', ')
                : 'Invalid data provided');
        throw Exception(message);
      }

      return response.data;
    } on DioException catch (e) {
      if (e.response?.statusCode == 400) {
        final errors = e.response?.data['errors'] ?? e.response?.data;
        final message = e.response?.data['message'] ??
            (errors is Map
                ? errors.values.join(', ')
                : 'Invalid data provided');
        throw Exception(message);
      }
      throw Exception('Failed to create journal: ${e.message}');
    }
  }

  Future<Map<String, dynamic>> updateJournal(
    int id,
    Map<String, dynamic> data,
  ) async {
    try {
      final response = await _dio.put(
        ApiConfig.getFullUrl('journal/$id/'),
        data: data,
        options: Options(
          validateStatus: (status) => status == 200 || status == 400,
        ),
      );

      return response.data;
    } catch (e) {
      throw Exception('Failed to update journal: $e');
    }
  }

  Future<void> deleteJournal(int id) async {
    try {
      final response = await _dio.delete(
        ApiConfig.getFullUrl('journal/$id/'),
        options: Options(
          validateStatus: (status) =>
              status == 204 || status == 404 || status == 400,
        ),
      );

      if (response.statusCode == 404) {
        throw Exception('Journal not found');
      }

      if (response.statusCode == 400) {
        final error = response.data['error'] ?? 'Failed to delete journal';
        throw Exception(error);
      }

      if (response.statusCode != 204) {
        throw Exception('Failed to delete journal');
      }
    } catch (e) {
      if (e is DioException && e.response?.statusCode == 404) {
        throw Exception('Journal not found');
      }
      throw Exception('Failed to delete journal: ${e.toString()}');
    }
  }

  Future<Map<String, dynamic>> analyzeSentiment(int id) async {
    try {
      final response = await _dio.get(
        ApiConfig.getFullUrl('journal/$id/analyze_sentiment/'),
        options: Options(validateStatus: (status) => status == 200),
      );

      if (response.data['status'] != 'success' ||
          response.data['data'] == null) {
        throw Exception('Invalid response format');
      }

      return response.data['data'];
    } catch (e) {
      print('Error analyzing sentiment: $e');
      throw Exception('Failed to analyze sentiment: $e');
    }
  }

  Future<List<Map<String, dynamic>>> analyzeAllSentiments() async {
    try {
      final response = await _dio.get(
        ApiConfig.getFullUrl(ApiConfig.analyzeAllSentimentsEndpoint),
        options: Options(validateStatus: (status) => status == 200),
      );

      if (response.data['status'] != 'success' ||
          response.data['data'] == null) {
        throw Exception('Invalid response format');
      }

      return List<Map<String, dynamic>>.from(response.data['data']);
    } catch (e) {
      debugPrint('Error analyzing all sentiments: $e');
      throw Exception('Failed to analyze all sentiments: $e');
    }
  }
}
