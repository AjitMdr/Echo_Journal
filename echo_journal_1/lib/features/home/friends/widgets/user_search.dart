import 'dart:async';
import 'package:flutter/material.dart';
import 'package:echo_journal_1/services/friends/friends_service.dart';
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';

class UserSearchWidget extends StatefulWidget {
  final bool isDarkMode;
  final VoidCallback? onFriendStatusChanged;

  const UserSearchWidget({
    super.key,
    required this.isDarkMode,
    this.onFriendStatusChanged,
  });

  @override
  State<UserSearchWidget> createState() => _UserSearchWidgetState();
}

class _UserSearchWidgetState extends State<UserSearchWidget> {
  final FriendsService _friendsService = FriendsService();
  final TextEditingController _searchController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  Timer? _debounce;
  List<Map<String, dynamic>> _searchResults = [];
  bool _isLoading = false;
  bool _hasSearched = false;
  bool _hasMoreData = true;
  int _currentPage = 1;
  String _lastSearchQuery = '';
  bool _isLoadingMore = false;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_scrollListener);
    _searchController.addListener(_onSearchChanged);
  }

  @override
  void dispose() {
    _searchController.removeListener(_onSearchChanged);
    _searchController.dispose();
    _scrollController.dispose();
    _debounce?.cancel();
    super.dispose();
  }

  void _onSearchChanged() {
    if (_debounce?.isActive ?? false) _debounce!.cancel();
    _debounce = Timer(const Duration(milliseconds: 500), () {
      _handleSearch();
    });
  }

  void _scrollListener() {
    if (_scrollController.position.pixels ==
        _scrollController.position.maxScrollExtent) {
      if (!_isLoading && _hasMoreData) {
        _loadMoreResults();
      }
    }
  }

  Future<void> _loadMoreResults() async {
    if (!_hasMoreData || _isLoadingMore || _lastSearchQuery.isEmpty) return;

    setState(() => _isLoadingMore = true);

    try {
      final results = await _friendsService.searchUsers(
        _lastSearchQuery,
        page: _currentPage + 1,
      );

      if (mounted) {
        setState(() {
          _searchResults.addAll(results['results']);
          _currentPage++;
          _hasMoreData = results['next'] != null;
          _isLoadingMore = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoadingMore = false);
        ToastHelper.showError(context, 'Failed to load more results: $e');
      }
    }
  }

  Future<void> _handleSearch() async {
    final query = _searchController.text.trim();
    if (query.isEmpty) {
      setState(() {
        _searchResults = [];
        _hasSearched = false;
        _hasMoreData = true;
        _currentPage = 1;
        _lastSearchQuery = '';
      });
      return;
    }

    // If this is a refresh with the same query, don't show loading indicator
    final isRefresh = query == _lastSearchQuery;

    setState(() {
      if (!isRefresh) {
        _isLoading = true;
        _currentPage = 1;
      }
      _lastSearchQuery = query;
    });

    try {
      final results = await _friendsService.searchUsers(query, page: 1);
      if (mounted) {
        setState(() {
          _searchResults = results['results'];
          _hasSearched = true;
          _hasMoreData = results['next'] != null;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        ToastHelper.showError(context, 'Failed to search users: $e');
      }
    }
  }

  Future<void> _handleSendRequest(Map<String, dynamic> user) async {
    try {
      // Get the user ID and ensure it's an integer
      final userId = user['id'] is String ? int.parse(user['id']) : user['id'];

      print('🔄 Sending friend request to user ID: $userId'); // Debug log

      await _friendsService.sendFriendRequest(userId);
      if (mounted) {
        ToastHelper.showSuccess(
          context,
          'Friend request sent to ${user["username"]}',
        );

        // Update the user's friendship status in the current results
        setState(() {
          for (var i = 0; i < _searchResults.length; i++) {
            if (_searchResults[i]['id'] == userId) {
              _searchResults[i]['friendship_status'] = 'pending_sent';
              break;
            }
          }
        });

        // Also refresh the search to get updated data from the server
        _handleSearch();

        // Notify parent that friend status has changed
        widget.onFriendStatusChanged?.call();
      }
    } catch (e) {
      if (mounted) {
        String errorMessage = e.toString();
        // Remove the "Exception: " prefix if present
        if (errorMessage.startsWith('Exception: ')) {
          errorMessage = errorMessage.substring('Exception: '.length);
        }

        print('❌ Error sending friend request: $errorMessage'); // Debug log

        ToastHelper.showError(context, errorMessage);
      }
    }
  }

  Widget _buildUserTile(Map<String, dynamic> user) {
    return ListTile(
      leading: CircleAvatar(
        radius: 24,
        backgroundColor: Colors.blue.shade100,
        child: Text(
          user['username']?[0]?.toUpperCase() ?? '?',
          style: TextStyle(
            color: Colors.blue.shade900,
            fontWeight: FontWeight.bold,
            fontSize: 18,
          ),
        ),
      ),
      title: Text(
        user['username'] ?? '',
        style: const TextStyle(fontWeight: FontWeight.bold),
      ),
      subtitle: Text(user['email'] ?? ''),
      trailing: _buildActionButton(user),
    );
  }

  Widget _buildActionButton(Map<String, dynamic> user) {
    final status = user['friendship_status'] as String? ?? 'none';
    print(
      '🔘 Building action button for user ${user['username']} with status: $status',
    ); // Debug log

    switch (status) {
      case 'friend':
        print('💬 Showing chat button'); // Debug log
        return Container(
          height: 40,
          width: 40,
          decoration: BoxDecoration(
            color: Colors.blue.shade50,
            borderRadius: BorderRadius.circular(20),
          ),
          child: IconButton(
            icon: const Icon(Icons.chat_bubble_outline),
            color: Colors.blue,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            onPressed: () => _navigateToChat(user),
          ),
        );
      case 'pending_sent':
        print('⏳ Showing pending sent button with cancel option'); // Debug log
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              height: 40,
              width: 40,
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(20),
              ),
              child: IconButton(
                icon: const Icon(Icons.timer),
                color: Colors.grey,
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
                onPressed: null,
              ),
            ),
            SizedBox(width: 8),
            Container(
              height: 40,
              width: 40,
              decoration: BoxDecoration(
                color: Colors.red.shade50,
                borderRadius: BorderRadius.circular(20),
              ),
              child: IconButton(
                icon: const Icon(Icons.cancel_outlined),
                color: Colors.red,
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
                onPressed: () => _handleCancelRequest(user),
              ),
            ),
          ],
        );
      case 'pending_received':
        print('✅ Showing accept/reject buttons'); // Debug log
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            IconButton(
              onPressed: () => _handleRejectRequest(user),
              icon: Icon(Icons.close, color: Colors.red),
              style: IconButton.styleFrom(
                padding: EdgeInsets.zero,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                  side: BorderSide(color: Colors.red),
                ),
                minimumSize: Size(40, 40),
              ),
            ),
            SizedBox(width: 8),
            IconButton(
              onPressed: () => _handleAcceptRequest(user),
              icon: Icon(Icons.check, color: Colors.white),
              style: IconButton.styleFrom(
                backgroundColor: Colors.green,
                padding: EdgeInsets.zero,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                minimumSize: Size(40, 40),
              ),
            ),
          ],
        );
      case 'none':
      default:
        print('➕ Showing add friend button'); // Debug log
        return Container(
          height: 40,
          width: 40,
          decoration: BoxDecoration(
            color: Colors.blue.shade50,
            borderRadius: BorderRadius.circular(20),
          ),
          child: IconButton(
            icon: const Icon(Icons.person_add),
            color: Colors.blue,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            onPressed: () => _handleSendRequest(user),
          ),
        );
    }
  }

  void _navigateToChat(Map<String, dynamic> user) {
    Navigator.of(context).pushNamed(
      '/chat',
      arguments: {
        'userId': user['id'],
        'username': user['username'],
        'profileImage': user['profile_image'],
      },
    );
  }

  Future<void> _handleAcceptRequest(Map<String, dynamic> user) async {
    try {
      final userId = user['id'] is String ? int.parse(user['id']) : user['id'];

      print('🔄 Accepting friend request from user ID: $userId'); // Debug log

      await _friendsService.acceptFriendRequest(userId.toString());
      if (mounted) {
        ToastHelper.showSuccess(
          context,
          'Friend request from ${user["username"]} accepted',
        );

        // Update the user's friendship status in the current results
        setState(() {
          for (var i = 0; i < _searchResults.length; i++) {
            if (_searchResults[i]['id'] == userId) {
              _searchResults[i]['friendship_status'] = 'friend';
              break;
            }
          }
        });

        // Also refresh the search to get updated data from the server
        _handleSearch();

        // Notify parent that friend status has changed
        widget.onFriendStatusChanged?.call();
      }
    } catch (e) {
      if (mounted) {
        String errorMessage = e.toString();
        // Remove the "Exception: " prefix if present
        if (errorMessage.startsWith('Exception: ')) {
          errorMessage = errorMessage.substring('Exception: '.length);
        }

        print('❌ Error accepting friend request: $errorMessage'); // Debug log

        ToastHelper.showError(
          context,
          'Failed to accept request: $errorMessage',
        );
      }
    }
  }

  Future<void> _handleRejectRequest(Map<String, dynamic> user) async {
    try {
      final userId = user['id'] is String ? int.parse(user['id']) : user['id'];

      print('🔄 Rejecting friend request from user ID: $userId'); // Debug log

      await _friendsService.rejectFriendRequest(userId.toString());
      if (mounted) {
        ToastHelper.showInfo(
          context,
          'Friend request from ${user["username"]} rejected',
        );

        // Update the user's friendship status in the current results
        setState(() {
          for (var i = 0; i < _searchResults.length; i++) {
            if (_searchResults[i]['id'] == userId) {
              _searchResults[i]['friendship_status'] = 'none';
              break;
            }
          }
        });

        // Also refresh the search to get updated data from the server
        _handleSearch();

        // Notify parent that friend status has changed
        widget.onFriendStatusChanged?.call();
      }
    } catch (e) {
      if (mounted) {
        String errorMessage = e.toString();
        // Remove the "Exception: " prefix if present
        if (errorMessage.startsWith('Exception: ')) {
          errorMessage = errorMessage.substring('Exception: '.length);
        }

        print('❌ Error rejecting friend request: $errorMessage'); // Debug log

        ToastHelper.showError(
          context,
          'Failed to reject request: $errorMessage',
        );
      }
    }
  }

  Future<void> _handleCancelRequest(Map<String, dynamic> user) async {
    try {
      final userId = user['id'] is String ? int.parse(user['id']) : user['id'];
      await _friendsService.cancelFriendRequest(user['id'].toString());
      if (mounted) {
        ToastHelper.showInfo(
          context,
          'Friend request to ${user["username"]} canceled',
        );

        // Update the user's friendship status in the current results
        setState(() {
          for (var i = 0; i < _searchResults.length; i++) {
            if (_searchResults[i]['id'] == userId) {
              _searchResults[i]['friendship_status'] = 'none';
              break;
            }
          }
        });

        // Also refresh the search to get updated data from the server
        _handleSearch();

        // Notify parent that friend status has changed
        widget.onFriendStatusChanged?.call();
      }
    } catch (e) {
      if (mounted) {
        String errorMessage = e.toString();
        // Remove the "Exception: " prefix if present
        if (errorMessage.startsWith('Exception: ')) {
          errorMessage = errorMessage.substring('Exception: '.length);
        }

        print('❌ Error canceling friend request: $errorMessage'); // Debug log

        ToastHelper.showError(
          context,
          'Failed to cancel request: $errorMessage',
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: EdgeInsets.all(16),
          child: TextField(
            controller: _searchController,
            decoration: InputDecoration(
              hintText: 'Search users...',
              prefixIcon: Icon(Icons.search),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              contentPadding: EdgeInsets.symmetric(
                horizontal: 16,
                vertical: 12,
              ),
            ),
            onSubmitted: (_) => _handleSearch(),
          ),
        ),
        if (_isLoading && _searchResults.isEmpty)
          Expanded(child: Center(child: CircularProgressIndicator()))
        else if (_searchResults.isEmpty && _hasSearched)
          Expanded(
            child: AnimationConfiguration.synchronized(
              duration: const Duration(milliseconds: 500),
              child: SlideAnimation(
                verticalOffset: 50.0,
                child: FadeInAnimation(
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.search_off, size: 64, color: Colors.grey),
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
                          style: TextStyle(color: Colors.grey, fontSize: 14),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          )
        else if (_searchResults.isEmpty)
          Expanded(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.search, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    'Search for users',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.grey,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Enter a name or email to find users',
                    style: TextStyle(color: Colors.grey, fontSize: 14),
                  ),
                ],
              ),
            ),
          )
        else
          Expanded(
            child: AnimationLimiter(
              child: ListView.builder(
                controller: _scrollController,
                padding: EdgeInsets.all(16),
                itemCount: _searchResults.length + (_hasMoreData ? 1 : 0),
                itemBuilder: (context, index) {
                  if (index == _searchResults.length) {
                    return Center(
                      child: Padding(
                        padding: EdgeInsets.all(16),
                        child: CircularProgressIndicator(),
                      ),
                    );
                  }

                  final user = _searchResults[index];
                  return AnimationConfiguration.staggeredList(
                    position: index,
                    duration: const Duration(milliseconds: 375),
                    child: SlideAnimation(
                      verticalOffset: 50.0,
                      child: FadeInAnimation(
                        child: Card(
                          color: widget.isDarkMode
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
                              backgroundColor: Colors.purple.shade100,
                              child: Text(
                                user['username']?[0]?.toUpperCase() ?? '?',
                                style: TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.purple.shade900,
                                ),
                              ),
                            ),
                            title: Text(
                              user['username'] ?? 'Unknown',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                            subtitle: Text(
                              user['email'] ?? '',
                              style: TextStyle(
                                color: widget.isDarkMode
                                    ? Colors.grey[300]
                                    : Colors.grey[600],
                              ),
                            ),
                            trailing: _buildActionButton(user),
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
    );
  }
}
