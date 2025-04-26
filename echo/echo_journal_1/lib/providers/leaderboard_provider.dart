import 'package:flutter/material.dart';
import 'package:echo_journal1/services/streak/streak_service.dart';

class LeaderboardProvider extends ChangeNotifier {
  final StreakService _streakService;
  List<dynamic> _leaderboardData = [];
  String? _error;
  bool _isLoading = false;
  bool _hasNextPage = false;
  int _currentPage = 1;
  int _totalPages = 1;
  static const int _pageSize = 20;

  LeaderboardProvider(this._streakService);

  List<dynamic> get leaderboardData => _leaderboardData;
  String? get error => _error;
  bool get isLoading => _isLoading;
  bool get hasNextPage => _hasNextPage;
  int get currentPage => _currentPage;
  int get totalPages => _totalPages;

  Future<void> fetchLeaderboard() async {
    try {
      _isLoading = true;
      _error = null;
      notifyListeners();

      debugPrint('🔄 Fetching leaderboard with page: $_currentPage');

      final response = await _streakService.getLeaderboard(
        type: 'overall',
        page: _currentPage,
        pageSize: _pageSize,
      );

      if (response['results'] != null) {
        if (_currentPage == 1) {
          _leaderboardData = response['results'];
        } else {
          _leaderboardData = [..._leaderboardData, ...response['results']];
        }
        _totalPages = response['total_pages'] ?? 1;
        _hasNextPage = _currentPage < _totalPages;
      } else {
        _error = 'No data available';
      }
    } catch (e) {
      _error = e.toString();
      debugPrint('❌ Leaderboard error: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> loadNextPage() async {
    if (_hasNextPage && !_isLoading) {
      _currentPage++;
      await fetchLeaderboard();
    }
  }

  Future<void> refresh() async {
    _currentPage = 1;
    await fetchLeaderboard();
  }
}

class LeaderboardEntry {
  final int userId;
  final String username;
  final String? profileImage;
  final int streakCount;
  final String rankBadge;
  final bool isCurrentUser;

  LeaderboardEntry({
    required this.userId,
    required this.username,
    this.profileImage,
    required this.streakCount,
    required this.rankBadge,
    required this.isCurrentUser,
  });

  factory LeaderboardEntry.fromJson(Map<String, dynamic> json) {
    return LeaderboardEntry(
      userId: json['user_id'],
      username: json['username'],
      profileImage: json['profile_picture'],
      streakCount: json['streak_count'],
      rankBadge: json['rank_badge'],
      isCurrentUser: json['is_current_user'] ?? false,
    );
  }
}
