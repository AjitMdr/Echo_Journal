import 'dart:async';
import 'package:flutter/material.dart';
import 'package:echo_journal1/services/friends/friends_service.dart';
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/utils/toast_helper.dart';

class UsersPage extends StatefulWidget {
  const UsersPage({super.key});

  @override
  State<UsersPage> createState() => _UsersPageState();
}

class _UsersPageState extends State<UsersPage> {
  final FriendsService _friendsService = FriendsService();
  final TextEditingController _searchController = TextEditingController();
  List<Map<String, dynamic>> _users = [];
  bool _isLoading = false;
  Timer? _debounce;

  @override
  void initState() {
    super.initState();
    _searchUsers('');
  }

  @override
  void dispose() {
    _searchController.dispose();
    _debounce?.cancel();
    super.dispose();
  }

  Future<void> _searchUsers(String query) async {
    if (_debounce?.isActive ?? false) _debounce!.cancel();
    _debounce = Timer(const Duration(milliseconds: 500), () async {
      setState(() => _isLoading = true);
      try {
        final results = await _friendsService.searchUsers(query);
        if (mounted) {
          setState(() => _users = results['results']);
        }
      } catch (e) {
        if (mounted) {
          ToastHelper.showError(context, 'Failed to search users: $e');
        }
      } finally {
        if (mounted) {
          setState(() => _isLoading = false);
        }
      }
    });
  }

  Future<void> _handleSendRequest(Map<String, dynamic> user) async {
    setState(() => _isLoading = true);
    try {
      await _friendsService.sendFriendRequest(user['id'].toString());
      if (mounted) {
        ToastHelper.showSuccess(
          context,
          'Friend request sent to ${user['name']}',
        );
      }
      _searchUsers(_searchController.text);
    } catch (e) {
      if (mounted) {
        ToastHelper.showError(context, 'Failed to send friend request: $e');
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Container(
              color: Theme.of(context).scaffoldBackgroundColor,
              padding: EdgeInsets.all(16),
              child: Column(
                children: [
                  Row(
                    children: [
                      Icon(Icons.group, size: 24),
                      SizedBox(width: 8),
                      Text(
                        'All Users',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 16),
                  TextField(
                    controller: _searchController,
                    style: TextStyle(
                      color: isDarkMode ? Colors.white : Colors.black,
                    ),
                    decoration: InputDecoration(
                      hintText: 'Search users by name or email...',
                      hintStyle: TextStyle(
                        color: isDarkMode ? Colors.grey[400] : Colors.grey[600],
                      ),
                      prefixIcon: Icon(
                        Icons.search,
                        color: isDarkMode ? Colors.grey[400] : Colors.grey[600],
                      ),
                      suffixIcon: _searchController.text.isNotEmpty
                          ? IconButton(
                              icon: Icon(Icons.clear),
                              onPressed: () {
                                _searchController.clear();
                                _searchUsers('');
                              },
                            )
                          : null,
                      filled: true,
                      fillColor:
                          isDarkMode ? Colors.grey[800] : Colors.grey[200],
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide.none,
                      ),
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 14,
                      ),
                    ),
                    onChanged: _searchUsers,
                  ),
                ],
              ),
            ),
            Expanded(
              child: _isLoading
                  ? Center(child: CircularProgressIndicator())
                  : _users.isEmpty
                      ? Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.search_off,
                                size: 64,
                                color: Colors.grey,
                              ),
                              SizedBox(height: 16),
                              Text(
                                'No users found',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.grey,
                                ),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Try a different search term',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 14,
                                ),
                              ),
                            ],
                          ),
                        )
                      : AnimationLimiter(
                          child: ListView.builder(
                            padding: EdgeInsets.all(16),
                            itemCount: _users.length,
                            itemBuilder: (context, index) {
                              final user = _users[index];
                              return AnimationConfiguration.staggeredList(
                                position: index,
                                duration: const Duration(milliseconds: 375),
                                child: SlideAnimation(
                                  verticalOffset: 50.0,
                                  child: FadeInAnimation(
                                    child: Card(
                                      color: isDarkMode
                                          ? Colors.grey[800]
                                          : Colors.white,
                                      elevation: 2,
                                      margin: EdgeInsets.only(bottom: 8),
                                      shape: RoundedRectangleBorder(
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: ListTile(
                                        contentPadding: EdgeInsets.all(12),
                                        leading: CircleAvatar(
                                          radius: 24,
                                          backgroundColor:
                                              Colors.purple.shade100,
                                          child: Text(
                                            user['name']?[0]?.toUpperCase() ??
                                                '?',
                                            style: TextStyle(
                                              fontSize: 20,
                                              fontWeight: FontWeight.bold,
                                              color: Colors.purple.shade900,
                                            ),
                                          ),
                                        ),
                                        title: Text(
                                          user['name'] ?? 'Unknown',
                                          style: TextStyle(
                                            fontWeight: FontWeight.bold,
                                            fontSize: 16,
                                          ),
                                        ),
                                        subtitle: Column(
                                          crossAxisAlignment:
                                              CrossAxisAlignment.start,
                                          children: [
                                            SizedBox(height: 4),
                                            Text(
                                              user['email'] ?? '',
                                              style: TextStyle(
                                                color: isDarkMode
                                                    ? Colors.grey[300]
                                                    : Colors.grey[600],
                                              ),
                                            ),
                                          ],
                                        ),
                                        trailing: user['is_friend'] == true
                                            ? Chip(
                                                label: Text(
                                                  'Friend',
                                                  style: TextStyle(
                                                    color: Colors.white,
                                                    fontSize: 12,
                                                  ),
                                                ),
                                                backgroundColor: Colors.green,
                                              )
                                            : user['request_sent'] == true
                                                ? Chip(
                                                    label: Text(
                                                      'Pending',
                                                      style: TextStyle(
                                                        color: Colors.white,
                                                        fontSize: 12,
                                                      ),
                                                    ),
                                                    backgroundColor:
                                                        Colors.orange,
                                                  )
                                                : IconButton(
                                                    icon: Icon(
                                                      Icons.person_add,
                                                      color: Theme.of(
                                                        context,
                                                      ).primaryColor,
                                                    ),
                                                    onPressed: () =>
                                                        _handleSendRequest(
                                                      user,
                                                    ),
                                                  ),
                                      ),
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
            ),
          ],
        ),
      ),
    );
  }
}
