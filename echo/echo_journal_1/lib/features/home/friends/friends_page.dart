import 'package:flutter/material.dart';
import 'package:echo_journal1/services/friends/friends_service.dart';
import 'package:echo_journal1/features/home/friends/widgets/friend_card.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'package:echo_journal1/features/home/chat/direct_chat_page.dart';
import 'package:echo_journal1/features/home/friends/widgets/user_search.dart';
import 'package:echo_journal1/features/home/friends/widgets/friend_requests.dart';
import 'package:echo_journal1/features/home/friends/widgets/friends_list.dart';

class FriendsPage extends StatefulWidget {
  final bool isDarkMode;

  const FriendsPage({Key? key, this.isDarkMode = false}) : super(key: key);

  @override
  _FriendsPageState createState() => _FriendsPageState();
}

class _FriendsPageState extends State<FriendsPage> {
  final FriendsService _friendsService = FriendsService();
  List<Map<String, dynamic>> _friends = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadFriends();
  }

  Future<void> _loadFriends() async {
    try {
      setState(() => _isLoading = true);
      final friends = await _friendsService.getFriends();
      if (mounted) {
        setState(() {
          _friends = friends;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        ToastHelper.showError(context, 'Failed to load friends: $e');
      }
    }
  }

  void _navigateToChat(Map<String, dynamic> friend) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DirectChatPage(
          friendId: friend['id'].toString(),
          friendName: friend['username'] ?? 'Unknown',
          isDarkMode: widget.isDarkMode,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Column(
        children: [
          TabBar(
            tabs: [
              Tab(icon: Icon(Icons.people), text: 'Friends'),
              Tab(icon: Icon(Icons.person_add), text: 'Requests'),
              Tab(icon: Icon(Icons.search), text: 'Search'),
            ],
          ),
          Expanded(
            child: TabBarView(
              children: [
                FriendsListWidget(
                  isDarkMode: widget.isDarkMode,
                  onFriendStatusChanged: _loadFriends,
                ),
                FriendRequestsWidget(
                  isDarkMode: widget.isDarkMode,
                  onFriendStatusChanged: _loadFriends,
                ),
                UserSearchWidget(
                  isDarkMode: widget.isDarkMode,
                  onFriendStatusChanged: _loadFriends,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFriendsList() {
    if (_isLoading) {
      return Center(child: CircularProgressIndicator());
    }

    if (_friends.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.people_outline, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text(
              'No friends yet',
              style: TextStyle(fontSize: 18, color: Colors.grey),
            ),
            SizedBox(height: 8),
            Text(
              'Add friends to start chatting',
              style: TextStyle(color: Colors.grey),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadFriends,
      child: ListView.builder(
        itemCount: _friends.length,
        itemBuilder: (context, index) {
          final friend = _friends[index];
          return null;
        },
      ),
    );
  }

  Widget _buildRequestsList() {
    return Center(child: Text('Friend Requests Coming Soon'));
  }

  Widget _buildSearchView() {
    return Center(child: Text('Search Friends Coming Soon'));
  }
}
