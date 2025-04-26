class ApiConfig {
  static const String baseUrl = 'http://127.0.0.1:8000';

  static String getFullUrl(String endpoint) {
    return '$baseUrl/api/$endpoint';
  }
}
