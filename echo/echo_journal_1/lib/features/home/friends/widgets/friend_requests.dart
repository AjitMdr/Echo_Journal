import 'package:flutter/material.dart';
import 'package:echo_fe/services/friends/friends_service.dart';
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';
import 'package:echo_fe/utils/toast_helper.dart';
import 'package:echo_fe/services/auth/secure_storage_service.dart';

class FriendRequestsWidget extends StatefulWidget {
  final bool isDarkMode;
  final VoidCallback? onFriendStatusChanged;

  const FriendRequestsWidget({
    super.key,
    required this.isDarkMode,
    this.onFriendStatusChanged,
  });

  @override
  State<FriendRequestsWidget> createState() => _FriendRequestsWidgetState();
}

class _FriendRequestsWidgetState extends State<FriendRequestsWidget> {
  final FriendsService _friendsService = FriendsService();
  List<Map<String, dynamic>> _requests = [];
  bool _isLoading = false;
  String? _currentUserId;

  @override
  void initState() {
    super.initState();
    _loadCurrentUserId();
  }

  Future<void> _loadCurrentUserId() async {
    try {
      final userId = await SecureStorageService.getUserId();
      print('Current User ID: $userId'); // Debug print

      if (userId != null) {
        setState(() {
          _currentUserId = userId;
        });
        await _loadRequests();
      } else {
        print('No user ID found in SecureStorage'); // Debug print
      }
    } catch (e) {
      print('Error loading user ID: $e');
    }
  }

  Future<void> _loadRequests() async {
    if (_currentUserId == null) return;

    setState(() => _isLoading = true);
    try {
      print(
        '🔄 Loading friend requests for user ID: $_currentUserId',
      ); // Debug log

      final requests = await _friendsService.getFriendRequests();
      print('📥 Received ${requests.length} friend requests'); // Debug log

      // Filter only pending requests where the current user is the recipient
      final filteredRequests = requests.where((request) {
        final toUser = request['to_user'] as Map<String, dynamic>;
        final isPending = request['status'] == 'pending';
        final isRecipient = toUser['id'].toString() == _currentUserId;

        print(
          '👤 Request ID: ${request['id']} - Status: ${request['status']} - To User ID: ${toUser['id']} - Is Recipient: $isRecipient',
        ); // Debug log

        return isPending && isRecipient;
      }).toList();

      print(
        '✅ Filtered to ${filteredRequests.length} pending requests for current user',
      ); // Debug log

      setState(() => _requests = filteredRequests);
    } catch (e) {
      print('❌ Error loading friend requests: $e'); // Debug log
      if (mounted) {
        ToastHelper.showError(context, 'Failed to load friend requests: $e');
      }
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _handleAccept(Map<String, dynamic> request) async {
    setState(() => _isLoading = true);
    try {
      final requestId = request['id'].toString();
      print('🔄 Accepting friend request ID: $requestId'); // Debug log

      await _friendsService.acceptFriendRequest(requestId);
      if (mounted) {
        final fromUser = request['from_user'] as Map<String, dynamic>;
        ToastHelper.showSuccess(
          context,
          'Accepted friend request from ${fromUser['username']}',
        );

        // Reload the requests after accepting
        await _loadRequests();

        // Notify parent that friend status has changed
        widget.onFriendStatusChanged?.call();
      }
    } catch (e) {
      setState(() => _isLoading = false);
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

  Future<void> _handleReject(Map<String, dynamic> request) async {
    setState(() => _isLoading = true);
    try {
      final requestId = request['id'].toString();
      print('🔄 Rejecting friend request ID: $requestId'); // Debug log

      await _friendsService.rejectFriendRequest(requestId);
      if (mounted) {
        final fromUser = request['from_user'] as Map<String, dynamic>;
        ToastHelper.showInfo(
          context,
          'Rejected friend request from ${fromUser['username']}',
        );

        // Reload the requests after rejecting
        await _loadRequests();

        // Notify parent that friend status has changed
        widget.onFriendStatusChanged?.call();
      }
    } catch (e) {
      setState(() => _isLoading = false);
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

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Center(child: CircularProgressIndicator());
    }

    if (_requests.isEmpty) {
      return AnimationConfiguration.synchronized(
        duration: const Duration(milliseconds: 500),
        child: SlideAnimation(
          verticalOffset: 50.0,
          child: FadeInAnimation(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.person_add_disabled, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    'No friend requests',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.grey,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'When someone sends you a friend request,\nit will appear here',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey, fontSize: 14),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadRequests,
      child: AnimationLimiter(
        child: ListView.builder(
          padding: EdgeInsets.all(16),
          itemCount: _requests.length,
          itemBuilder: (context, index) {
            final request = _requests[index];
            final fromUser = request['from_user'] as Map<String, dynamic>;
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
                    child: ListTile(
                      contentPadding: EdgeInsets.all(12),
                      leading: CircleAvatar(
                        radius: 24,
                        backgroundColor: Colors.blue.shade100,
                        child: Text(
                          fromUser['username'][0].toUpperCase(),
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: Colors.blue.shade900,
                          ),
                        ),
                      ),
                      title: Text(
                        fromUser['username'],
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      subtitle: Text(
                        fromUser['email'],
                        style: TextStyle(
                          color: widget.isDarkMode
                              ? Colors.grey[300]
                              : Colors.grey[600],
                        ),
                      ),
                      trailing: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          IconButton(
                            onPressed: () => _handleReject(request),
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
                            onPressed: () => _handleAccept(request),
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
                      ),
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
