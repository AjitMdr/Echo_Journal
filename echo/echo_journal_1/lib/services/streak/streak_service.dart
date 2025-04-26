import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';
import 'package:flutter/foundation.dart';
import 'package:echo_journal1/core/configs/api_config.dart';

class StreakService {
  static const String authPrefix = '/auth';

  Future<Map<String, dynamic>> getCurrentStreak() async {
    try {
      final token = await SecureStorageService.getAccessToken();

      if (token == null) {
        debugPrint('❌ No authentication token found');
        return _getDefaultStreakData();
      }

      final response = await http.get(
        Uri.parse(
            '${ApiConfig.getFullUrl('$authPrefix/streaks/current_streak/')}'),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
      );

      debugPrint('🔄 Streak response status: ${response.statusCode}');
      debugPrint('🔄 Streak response body: ${response.body}');

      if (response.statusCode == 200) {
        final data = json.decode(utf8.decode(response.bodyBytes));
        debugPrint('🔄 Raw emoji from response: ${data['emoji']}');
        debugPrint('🔄 Raw response data: $data');

        // Ensure emoji is properly decoded
        String emoji = data['emoji'] ?? '💫';
        if (emoji.contains('ð') || emoji.contains('')) {
          // If emoji is corrupted, use default based on streak count
          final streakCount = data['current_streak'] ?? 0;
          emoji = _getDefaultEmoji(streakCount);
        }

        final streakData = {
          'current_streak': data['current_streak'] ?? 0,
          'longest_streak': data['longest_streak'] ?? 0,
          'last_journal_date': data['last_journal_date'],
          'emoji': emoji,
        };
        debugPrint('🔄 Processed streak data: $streakData');
        return streakData;
      } else if (response.statusCode == 404) {
        debugPrint('⚠️ No streak found, returning default values');
        return _getDefaultStreakData();
      } else {
        debugPrint('❌ Failed to get streak: ${response.statusCode}');
        return _getDefaultStreakData();
      }
    } catch (e) {
      debugPrint('❌ Error getting streak data: $e');
      return _getDefaultStreakData();
    }
  }

  Map<String, dynamic> _getDefaultStreakData() {
    return {
      'current_streak': 0,
      'longest_streak': 0,
      'last_journal_date': null,
      'emoji': '💫',
    };
  }

  String _getDefaultEmoji(int streakCount) {
    if (streakCount == 0) return '💫';
    if (streakCount < 7) return '🔥';
    if (streakCount < 30) return '⚡';
    if (streakCount < 100) return '🌟';
    return '👑';
  }

  Future<List<Map<String, dynamic>>> getUserBadges() async {
    try {
      final token = await SecureStorageService.getAccessToken();

      if (token == null) {
        debugPrint('❌ No authentication token found');
        return [];
      }

      final response = await http.get(
        Uri.parse('${ApiConfig.getFullUrl('$authPrefix/badges/user_badges/')}'),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return List<Map<String, dynamic>>.from(data);
      } else {
        debugPrint('❌ Failed to get badges: ${response.statusCode}');
        return [];
      }
    } catch (e) {
      debugPrint('❌ Error getting badges: $e');
      return [];
    }
  }

  Future<Map<String, dynamic>> getUserStreak(int userId) async {
    try {
      final token = await SecureStorageService.getAccessToken();

      if (token == null) {
        debugPrint('❌ No authentication token found');
        return _getDefaultStreakData();
      }

      debugPrint('🔄 Getting streak for user ID: $userId');
      final response = await http.get(
        Uri.parse(
            '${ApiConfig.getFullUrl('$authPrefix/streaks/user-streak/?user_id=$userId')}'),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      );

      debugPrint('🔄 User streak response status: ${response.statusCode}');
      debugPrint('🔄 User streak response body: ${response.body}');

      if (response.statusCode == 200) {
        final data = json.decode(utf8.decode(response.bodyBytes));
        debugPrint('🔄 Raw emoji from response: ${data['emoji']}');
        debugPrint('🔄 Raw response data: $data');

        // Ensure emoji is properly decoded
        String emoji = data['emoji'] ?? '💫';
        if (emoji.contains('ð') || emoji.contains('')) {
          // If emoji is corrupted, use default based on streak count
          final streakCount = data['current_streak'] ?? 0;
          emoji = _getDefaultEmoji(streakCount);
        }

        final streakData = {
          'current_streak': data['current_streak'] ?? 0,
          'longest_streak': data['longest_streak'] ?? 0,
          'last_journal_date': data['last_journal_date'],
          'emoji': emoji,
        };
        debugPrint('🔄 Processed streak data: $streakData');
        return streakData;
      } else if (response.statusCode == 404) {
        debugPrint(
          '⚠️ No streak found for user $userId, returning default values',
        );
        return _getDefaultStreakData();
      } else {
        debugPrint(
          '❌ Failed to get streak for user $userId: ${response.statusCode}',
        );
        return _getDefaultStreakData();
      }
    } catch (e) {
      debugPrint('❌ Error getting streak data for user $userId: $e');
      return _getDefaultStreakData();
    }
  }

  Future<Map<String, dynamic>> getLeaderboard({
    String type = 'overall',
    int page = 1,
    int pageSize = 10,
  }) async {
    try {
      debugPrint('🔄 Getting leaderboard data from API');
      final token = await SecureStorageService.getAccessToken();
      final isLoggedIn = await SecureStorageService.isLoggedIn();

      if (!isLoggedIn || token == null) {
        debugPrint('❌ User not logged in or no token found');
        throw Exception('Please log in to view the leaderboard');
      }

      // Build URL with query parameters
      final queryParams = {
        'type': type,
        'page': page.toString(),
        'page_size': pageSize.toString(),
      };

      final uri =
          Uri.parse(ApiConfig.getFullUrl('$authPrefix/streaks/leaderboard/'))
              .replace(queryParameters: queryParams);

      debugPrint('🔄 Making API request to: $uri');

      final response = await http.get(
        uri,
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
      );

      debugPrint('🔄 Response status: ${response.statusCode}');
      debugPrint('🔄 Response body: ${response.body}');

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data;
      } else {
        debugPrint('❌ Failed to get leaderboard: ${response.statusCode}');
        throw Exception('Failed to load leaderboard data');
      }
    } catch (e) {
      debugPrint('❌ Error getting leaderboard data: $e');
      throw Exception('Failed to load leaderboard data: $e');
    }
  }
}
