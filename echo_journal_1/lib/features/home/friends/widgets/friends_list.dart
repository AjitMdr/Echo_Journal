import 'package:flutter/material.dart';
import 'package:echo_journal_1/services/friends/friends_service.dart';
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'package:echo_journal_1/features/home/chat/direct_chat_page.dart';
import 'package:echo_journal_1/features/widgets/streak_badge_widget.dart';
import 'package:echo_journal_1/services/auth/secure_storage_service.dart';

class FriendsListWidget extends StatefulWidget {
  final bool isDarkMode;
  final Function()? onFriendStatusChanged;

  const FriendsListWidget({
    super.key,
    required this.isDarkMode,
    this.onFriendStatusChanged,
  });

  @override
  State<FriendsListWidget> createState() => _FriendsListWidgetState();
}

class _FriendsListWidgetState extends State<FriendsListWidget> {
  final FriendsService _friendsService = FriendsService();
  List<Map<String, dynamic>> _friends = [];
  bool _isLoading = false;
  String? _currentUserId;

  @override
  void initState() {
    super.initState();
    _loadCurrentUserId();
  }

  Future<void> _loadCurrentUserId() async {
    if (!mounted) return;

    setState(() => _isLoading = true);
    try {
      final userId = await SecureStorageService.getUserId();
      print('Current User ID: $userId'); // Debug print

      if (!mounted) return;

      if (userId != null) {
        setState(() {
          _currentUserId = userId;
        });

        await _loadFriends();
      } else {
        print('No user ID found in SecureStorage'); // Debug print
      }
    } catch (e) {
      print('Error loading user ID: $e'); // Debug print
      if (!mounted) return;
      ToastHelper.showError(context, 'Error loading user data: $e');
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _loadFriends() async {
    if (!mounted) return;

    setState(() => _isLoading = true);

    try {
      final friends = await _friendsService.getFriends();
      if (!mounted) return;

      setState(() {
        _friends = friends;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      ToastHelper.showError(context, 'Failed to load friends: $e');
    }
  }

  Future<void> _handleRemoveFriend(Map<String, dynamic> friend) async {
    if (!mounted) return;

    setState(() => _isLoading = true);
    try {
      final friendId = friend['id'].toString();
      final username = friend['username'] as String;
      print(
        '🔄 Removing friend ID: $friendId, Username: $username',
      ); // Debug log

      await _friendsService.removeFriend(friendId);
      if (!mounted) return;

      ToastHelper.showInfo(context, 'Removed $username from your friends');

      // Update the local list immediately
      setState(() {
        _friends.removeWhere((f) => f['id'].toString() == friendId);
      });

      // Notify parent that friend status has changed
      widget.onFriendStatusChanged?.call();
    } catch (e) {
      if (!mounted) return;

      String errorMessage = e.toString();
      // Remove the "Exception: " prefix if present
      if (errorMessage.startsWith('Exception: ')) {
        errorMessage = errorMessage.substring('Exception: '.length);
      }

      print('❌ Error removing friend: $errorMessage'); // Debug log
      ToastHelper.showError(context, 'Failed to remove friend: $errorMessage');
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  void _openDirectChat(Map<String, dynamic> friend) {
    final friendId = friend['id'].toString();
    final friendName = friend['username'] ?? friend['name'] ?? 'Unknown';

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DirectChatPage(
          friendId: friendId,
          friendName: friendName,
          isDarkMode: widget.isDarkMode,
        ),
      ),
    );
  }

  // Public method to refresh friends list
  void refreshFriends() {
    _loadFriends();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Center(child: CircularProgressIndicator());
    }

    if (_currentUserId == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 64, color: Colors.red),
            SizedBox(height: 16),
            Text(
              'Error: User not logged in',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.red,
              ),
            ),
            SizedBox(height: 8),
            ElevatedButton(onPressed: _loadCurrentUserId, child: Text('Retry')),
          ],
        ),
      );
    }

    if (_friends.isEmpty) {
      return AnimationConfiguration.synchronized(
        duration: const Duration(milliseconds: 500),
        child: SlideAnimation(
          verticalOffset: 50.0,
          child: FadeInAnimation(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.people_outline, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    'No friends yet',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.grey,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Search for users to add friends',
                    style: TextStyle(color: Colors.grey, fontSize: 14),
                  ),
                  SizedBox(height: 16),
                  ElevatedButton.icon(
                    onPressed: _loadFriends,
                    icon: Icon(Icons.refresh),
                    label: Text('Refresh'),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadFriends,
      child: AnimationLimiter(
        child: ListView.builder(
          padding: EdgeInsets.all(16),
          itemCount: _friends.length,
          itemBuilder: (context, index) {
            final friend = _friends[index];
            final DateTime createdDate = DateTime.parse(friend['created_at']);
            final String formattedDate =
                'Friends since ${createdDate.toLocal().toString().split(' ')[0]}';

            // Get the display name - prefer username, fallback to name
            final String displayName =
                friend['username'] ?? friend['name'] ?? 'Unknown';
            // Get the first letter for the avatar
            final String avatarLetter =
                displayName.isNotEmpty ? displayName[0].toUpperCase() : '?';

            return AnimationConfiguration.staggeredList(
              position: index,
              duration: const Duration(milliseconds: 375),
              child: SlideAnimation(
                verticalOffset: 50.0,
                child: FadeInAnimation(
                  child: Card(
                    color: widget.isDarkMode ? Colors.grey[800] : Colors.white,
                    elevation: 2,
                    margin: EdgeInsets.only(bottom: 8),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      children: [
                        ListTile(
                          contentPadding: EdgeInsets.all(12),
                          leading: Hero(
                            tag: 'friend_avatar_${friend['id']}',
                            child: CircleAvatar(
                              radius: 24,
                              backgroundColor: Colors.blue.shade100,
                              child: Text(
                                avatarLetter,
                                style: TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.blue.shade900,
                                ),
                              ),
                            ),
                          ),
                          title: Text(
                            displayName,
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                          trailing: IconButton(
                            icon: Icon(Icons.chat),
                            onPressed: () => _openDirectChat(friend),
                          ),
                        ),
                        // Streak display for friend
                        Padding(
                          padding: EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 8,
                          ),
                          child: StreakBadgeWidget(
                            userId: int.tryParse(friend['id']),
                            showBadges: false,
                            isDarkMode: widget.isDarkMode,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
