import 'package:flutter/foundation.dart';

class ApiConfig {
  static const bool useLocalHost = true; // Set to false for production

  // Base URLs with /api prefix
  static String get baseUrl {
    if (useLocalHost) {
      if (kIsWeb) {
        return 'http://localhost:8000/api'; // For web
      }
      return 'http://192.168.1.65:8000/api'; // For Android emulator
    }
    return 'https://production-api.com/api'; // Production URL
  }

  static String get wsBaseUrl {
    if (useLocalHost) {
      if (kIsWeb) {
        return 'ws://localhost:8000'; // For web
      }
      return 'ws://192.168.1.65:8000'; // For Android emulator
    }
    return 'wss://production-api.com'; // Production URL
  }

  // Auth endpoints (without leading slash)
  static String get tokenEndpoint => '/auth/login/';
  static String get tokenRefreshEndpoint => '/auth/token/refresh/';
  static String get tokenVerifyEndpoint => '/auth/token/verify/';

  // Chat endpoints (without leading or trailing slash)
  static String get chatEndpoints => '/direct-chat';
  static String get conversationsEndpoint => '$chatEndpoints/conversations';
  static String get messagesEndpoint => '$chatEndpoints/messages';
  static String get chatHistoryEndpoint => '$chatEndpoints/history';
  static String get unreadCountEndpoint =>
      '$conversationsEndpoint/unread_count';
  static String get recentConversationsEndpoint =>
      '$conversationsEndpoint/recent';
  static String get markMessagesAsReadEndpoint =>
      '$conversationsEndpoint/mark_read';

  // Journal endpoints (without leading slash)
  static String get journalEndpoint => 'journal';
  static String get journalListEndpoint => '$journalEndpoint?page=1';
  static String get analyzeAllSentimentsEndpoint =>
      '$journalEndpoint/analyze_all_sentiments';

  // Helper method to construct full URLs
  static String getFullUrl(String endpoint) {
    // Remove any leading slashes from the endpoint
    while (endpoint.startsWith('/')) {
      endpoint = endpoint.substring(1);
    }

    // Remove any trailing slashes from the endpoint
    while (endpoint.endsWith('/')) {
      endpoint = endpoint.substring(0, endpoint.length - 1);
    }

    // Ensure there's exactly one slash between baseUrl and endpoint
    final cleanBaseUrl = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;

    // For endpoints with query parameters, don't add trailing slash
    if (endpoint.contains('?')) {
      return '$cleanBaseUrl/$endpoint';
    }

    // Add trailing slash for all other endpoints
    return '$cleanBaseUrl/$endpoint/';
  }
}
