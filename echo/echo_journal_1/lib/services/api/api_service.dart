import 'package:dio/dio.dart';
import 'package:echo_fe/services/auth/secure_storage_service.dart';
import 'package:echo_fe/services/auth/session_management.dart';

class ApiService {
  final Dio _dio = Dio();

  ApiService() {
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          final token = await SecureStorageService.getAccessToken();
          if (token != null) {
            options.headers['Authorization'] = 'Bearer $token';
          }
          return handler.next(options);
        },
        onError: (DioException e, handler) async {
          if (e.response?.statusCode == 401) {
            // Try to refresh the token
            final refreshToken = await SecureStorageService.getRefreshToken();
            if (refreshToken != null) {
              final success =
                  await SessionManager.refreshAccessToken(refreshToken);
              if (success) {
                // Retry the original request with the new token
                final token = await SecureStorageService.getAccessToken();
                if (token != null) {
                  e.requestOptions.headers['Authorization'] = 'Bearer $token';
                  return handler.resolve(await _dio.fetch(e.requestOptions));
                }
              }
            }
            // If refresh failed or no refresh token, clear auth data
            await SecureStorageService.clearAuthData();
          }
          return handler.next(e);
        },
      ),
    );
  }

  Future<Response> get(String url) async {
    try {
      url = _sanitizeUrl(url);
      return await _dio.get(url);
    } catch (e) {
      rethrow;
    }
  }

  Future<Response> post(String url, {dynamic data}) async {
    try {
      url = _sanitizeUrl(url);
      return await _dio.post(url, data: data);
    } catch (e) {
      rethrow;
    }
  }

  Future<Response> put(String url, {dynamic data}) async {
    try {
      url = _sanitizeUrl(url);
      return await _dio.put(url, data: data);
    } catch (e) {
      rethrow;
    }
  }

  Future<Response> delete(String url) async {
    try {
      url = _sanitizeUrl(url);
      return await _dio.delete(url);
    } catch (e) {
      rethrow;
    }
  }

  String _sanitizeUrl(String url) {
    while (url.contains('/api/api/')) {
      url = url.replaceAll('/api/api/', '/api/');
    }

    if (url.startsWith('/apihttp')) {
      url = url.replaceFirst('/apihttp', 'http');
    }

    return url;
  }
}
