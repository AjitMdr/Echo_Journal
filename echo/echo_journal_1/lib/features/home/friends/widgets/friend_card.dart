import 'package:flutter/material.dart';
import 'package:echo_fe/services/streak/streak_service.dart';
import 'package:echo_fe/features/widgets/streak_badge_widget.dart';

class FriendCard extends StatefulWidget {
  final Map<String, dynamic> friend;
  final Function() onChatTap;
  final Function(Map<String, dynamic>) onUnfriend;
  final bool isDarkMode;

  const FriendCard({
    super.key,
    required this.friend,
    required this.onChatTap,
    required this.onUnfriend,
    this.isDarkMode = false,
  });

  @override
  State<FriendCard> createState() => _FriendCardState();
}

class _FriendCardState extends State<FriendCard> {
  final StreakService _streakService = StreakService();
  Map<String, dynamic>? _streakData;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadStreak();
  }

  Future<void> _loadStreak() async {
    try {
      final userId = int.parse(widget.friend['id'].toString());
      final streakData = await _streakService.getUserStreak(userId);
      if (mounted) {
        setState(() {
          _streakData = {
            'current_streak': streakData['current_streak'] ?? 0,
            'longest_streak': streakData['longest_streak'] ?? 0,
            'emoji': streakData['emoji'] ?? '💫',
          };
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _streakData = {
            'current_streak': 0,
            'longest_streak': 0,
            'emoji': '💫',
          };
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _showUnfriendDialog() async {
    final username = widget.friend['username'] ?? 'Unknown';
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Unfriend $username?'),
        content: Text(
            'Are you sure you want to remove $username from your friends list?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              widget.onUnfriend(widget.friend);
            },
            style: TextButton.styleFrom(
              foregroundColor: Colors.red,
            ),
            child: Text('Unfriend'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final username = widget.friend['username'] ?? 'Unknown';
    final firstLetter = username.isNotEmpty ? username[0].toUpperCase() : '?';

    return Card(
      elevation: 2,
      margin: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          ListTile(
            contentPadding: EdgeInsets.all(16),
            leading: CircleAvatar(
              radius: 28,
              backgroundColor: Colors.blue.shade100,
              child: Text(
                firstLetter,
                style: TextStyle(
                  color: Colors.blue.shade900,
                  fontWeight: FontWeight.bold,
                  fontSize: 20,
                ),
              ),
            ),
            title: Text(
              username,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
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
                    icon: Icon(Icons.person_remove, color: Colors.red),
                    onPressed: _showUnfriendDialog,
                  ),
                ),
                SizedBox(width: 8),
                Container(
                  decoration: BoxDecoration(
                    color: Colors.blue.shade50,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: IconButton(
                    icon: Icon(Icons.chat_bubble_outline, color: Colors.blue),
                    onPressed: widget.onChatTap,
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: StreakBadgeWidget(
              userId: int.parse(widget.friend['id'].toString()),
              showBadges: false,
              isDarkMode: widget.isDarkMode,
            ),
          ),
        ],
      ),
    );
  }
}
