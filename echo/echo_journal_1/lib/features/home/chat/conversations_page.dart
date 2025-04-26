import 'package:echo_journal1/models/chat/conversation.dart';
import 'package:echo_journal1/features/home/chat/direct_chat_page.dart';
import 'package:echo_journal1/services/chat/chat_service.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'package:flutter/material.dart';
import 'package:timeago/timeago.dart' as timeago;

class ConversationsPage extends StatefulWidget {
  final bool isDarkMode;

  const ConversationsPage({super.key, required this.isDarkMode});

  @override
  State<ConversationsPage> createState() => _ConversationsPageState();
}

class _ConversationsPageState extends State<ConversationsPage> {
  final ChatService _chatService = ChatService();
  List<Conversation> _conversations = [];
  bool _isLoading = false;
  bool _hasError = false;
  String? _errorMessage;
  String? _currentUserId;

  @override
  void initState() {
    super.initState();
    _getCurrentUserId();
    _loadConversations();
  }

  Future<void> _getCurrentUserId() async {
    final userId = await SecureStorageService.getUserId();
    if (userId != null) {
      setState(() {
        _currentUserId = userId;
      });
    }
  }

  Future<void> _loadConversations() async {
    if (mounted) {
      setState(() {
        _isLoading = true;
        _hasError = false;
        _errorMessage = null;
      });
    }

    try {
      final conversationsData = await _chatService.getRecentConversations();

      // Handle empty response
      if (conversationsData.isEmpty) {
        if (mounted) {
          setState(() {
            _conversations = [];
            _isLoading = false;
          });
        }
        return;
      }

      // Sort conversations by updated_at timestamp (most recent first)
      conversationsData.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));

      if (mounted) {
        setState(() {
          _conversations = conversationsData;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = e.toString();
        });
        ToastHelper.showError(context, 'Failed to load conversations: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final backgroundColor = widget.isDarkMode ? Colors.black : Colors.white;
    final textColor = widget.isDarkMode ? Colors.white : Colors.black;

    return Scaffold(
      backgroundColor: backgroundColor,
      body: Column(
        children: [
          // Error banner
          if (_hasError && _errorMessage != null)
            Container(
              padding: EdgeInsets.symmetric(vertical: 4, horizontal: 12),
              color: Colors.red,
              child: Row(
                children: [
                  Icon(Icons.error_outline, color: Colors.white, size: 18),
                  SizedBox(width: 4),
                  Expanded(
                    child: Text(
                      'Error loading conversations',
                      style: TextStyle(color: Colors.white, fontSize: 12),
                    ),
                  ),
                  TextButton(
                    onPressed: _loadConversations,
                    style: TextButton.styleFrom(
                      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 0),
                      minimumSize: Size(40, 24),
                    ),
                    child: Text(
                      'RETRY',
                      style: TextStyle(color: Colors.white, fontSize: 12),
                    ),
                  ),
                ],
              ),
            ),

          // Content
          Expanded(
            child: _isLoading
                ? Center(child: CircularProgressIndicator())
                : _conversations.isEmpty
                    ? _buildEmptyState()
                    : RefreshIndicator(
                        onRefresh: _loadConversations,
                        child: ListView.builder(
                          key: Key(
                              'conversations_list_${_conversations.length}'),
                          itemCount: _conversations.length,
                          itemBuilder: (context, index) {
                            final conversation = _conversations[index];
                            return _buildConversationTile(conversation);
                          },
                        ),
                      ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.chat_bubble_outline, size: 48, color: Colors.grey),
          SizedBox(height: 8),
          Text(
            'No conversations yet',
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 4),
          Text(
            'Start chatting with your friends!',
            style: TextStyle(fontSize: 12, color: Colors.grey),
          ),
          if (_hasError)
            Padding(
              padding: const EdgeInsets.only(top: 16.0),
              child: ElevatedButton.icon(
                onPressed: _loadConversations,
                icon: Icon(Icons.refresh, size: 16),
                label: Text('Retry', style: TextStyle(fontSize: 12)),
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  minimumSize: Size(80, 30),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildConversationTile(Conversation conversation) {
    if (_currentUserId == null) return SizedBox.shrink();

    final otherParticipant = conversation.getOtherParticipant(_currentUserId!);
    final lastMessage = conversation.lastMessage;
    final hasUnread = conversation.unreadCount > 0;
    final textColor = widget.isDarkMode ? Colors.white : Colors.black;

    return Card(
      margin: EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      elevation: 0.5,
      color: widget.isDarkMode ? Colors.grey[900] : Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: hasUnread
            ? BorderSide(color: Colors.purple.withOpacity(0.5), width: 1)
            : BorderSide.none,
      ),
      child: InkWell(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => DirectChatPage(
                friendId: otherParticipant.id,
                friendName: otherParticipant.username,
                isDarkMode: widget.isDarkMode,
                conversationId: conversation.id,
              ),
            ),
          ).then((_) => _loadConversations());
        },
        child: Padding(
          padding: EdgeInsets.all(12),
          child: Row(
            children: [
              // Profile Picture or Initial
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: widget.isDarkMode
                      ? Colors.purple.withOpacity(0.3)
                      : Colors.purple.withOpacity(0.1),
                ),
                child: otherParticipant.profilePictureUrl != null
                    ? ClipOval(
                        child: Image.network(
                          otherParticipant.profilePictureUrl!,
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) => Center(
                            child: Text(
                              otherParticipant.username[0].toUpperCase(),
                              style: TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.w500,
                                color: widget.isDarkMode
                                    ? Colors.white
                                    : Colors.purple,
                              ),
                            ),
                          ),
                        ),
                      )
                    : Center(
                        child: Text(
                          otherParticipant.username[0].toUpperCase(),
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w500,
                            color: widget.isDarkMode
                                ? Colors.white
                                : Colors.purple,
                          ),
                        ),
                      ),
              ),
              SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          otherParticipant.username,
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight:
                                hasUnread ? FontWeight.w600 : FontWeight.w500,
                            color: textColor,
                          ),
                        ),
                        if (lastMessage != null)
                          Text(
                            timeago.format(lastMessage.timestamp),
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey,
                            ),
                          ),
                      ],
                    ),
                    SizedBox(height: 4),
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            lastMessage?.content ?? 'No messages yet',
                            style: TextStyle(
                              fontSize: 14,
                              color: hasUnread ? textColor : Colors.grey,
                              fontWeight: hasUnread
                                  ? FontWeight.w500
                                  : FontWeight.normal,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        if (hasUnread)
                          Container(
                            margin: EdgeInsets.only(left: 8),
                            padding: EdgeInsets.symmetric(
                                horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: Colors.purple,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              conversation.unreadCount.toString(),
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
