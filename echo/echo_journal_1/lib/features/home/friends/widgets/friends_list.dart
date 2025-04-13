import 'package:flutter/material.dart';
import 'package:echo_journal1/services/friends/friends_service.dart';
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'package:echo_journal1/features/home/chat/direct_chat_page.dart';
import 'package:echo_journal1/features/widgets/streak_badge_widget.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';

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
      final response = await _friendsService.getFriends();
      setState(() {
        _friends = response;
      });
    } catch (e) {
      print('Error loading friends: $e');
      // Show error message
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load friends')),
      );
    } finally {
      if (!mounted) return;
      setState(() => _isLoading = false);
    }
  }

  Future<void> _handleRemoveFriend(Map<String, dynamic> friend) async {
    // Show confirmation dialog
    final bool? confirm = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Unfriend Confirmation'),
          content: Text(
              'Are you sure you want to unfriend ${friend['username'] ?? friend['name'] ?? 'this user'}?'),
          actions: <Widget>[
            TextButton(
              child: Text('Cancel'),
              onPressed: () => Navigator.of(context).pop(false),
            ),
            TextButton(
              child: Text('Unfriend'),
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              onPressed: () => Navigator.of(context).pop(true),
            ),
          ],
        );
      },
    );

    if (confirm == true) {
      try {
        // Get the friendship ID
        final friendshipId = friend['friendship_id'];
        if (friendshipId == null || friendshipId.isEmpty) {
          throw Exception('Friendship ID not found');
        }

        await _friendsService.unfriend(friendshipId);
        // Remove friend from local list
        setState(() {
          _friends.removeWhere((f) => f['id'] == friend['id']);
        });
        // Show success message
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Friend removed successfully')),
          );
        }
      } catch (e) {
        print('Error unfriending: $e');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Failed to remove friend')),
          );
        }
      }
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
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                decoration: BoxDecoration(
                                  color: Colors.red.shade50,
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: IconButton(
                                  icon: Icon(Icons.person_remove,
                                      color: Colors.red),
                                  onPressed: () => _handleRemoveFriend(friend),
                                ),
                              ),
                              SizedBox(width: 8),
                              Container(
                                decoration: BoxDecoration(
                                  color: Colors.blue.shade50,
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: IconButton(
                                  icon: Icon(Icons.chat_bubble_outline,
                                      color: Colors.blue),
                                  onPressed: () => _openDirectChat(friend),
                                ),
                              ),
                            ],
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
