import 'dart:async';
import 'package:echo_journal_1/data/models/direct_message.dart';
import 'package:echo_journal_1/features/home/chat/widgets/message_bubble.dart';
import 'package:echo_journal_1/services/chat/chat_service.dart';
import 'package:echo_journal_1/services/auth/secure_storage_service.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'package:flutter/material.dart';
import 'package:echo_journal_1/core/configs/api_config.dart';

class DirectChatPage extends StatefulWidget {
  final String friendId;
  final String friendName;
  final bool isDarkMode;
  final String? conversationId;

  const DirectChatPage({
    super.key,
    required this.friendId,
    required this.friendName,
    required this.isDarkMode,
    this.conversationId,
  });

  @override
  State<DirectChatPage> createState() => _DirectChatPageState();
}

class _DirectChatPageState extends State<DirectChatPage> {
  final ChatService _chatService = ChatService();
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  List<DirectMessage> _messages = [];
  bool _isLoading = false;
  bool _isConnecting = false;
  bool _hasConnectionError = false;
  String? _connectionError;
  String? _currentUsername;
  String? _currentUserId;
  String? _conversationId;
  StreamSubscription? _messageSubscription;
  bool _isSending = false;
  bool _hasUnreadMessages = true;
  String? _loadError;
  bool _isConnected = false;
  Timer? _reconnectionTimer;
  int _retryAttempt = 0;

  @override
  void initState() {
    super.initState();

    // Debug the original friendId to understand where the hash comes from
    debugPrint('🔍 ORIGINAL friendId from widget: "${widget.friendId}"');
    debugPrint('🔍 Widget object info: ${widget.toString()}');

    _getCurrentUser();
    _conversationId = widget.conversationId;
    _loadChatHistory();

    // Mark messages as read when chat is opened
    _markMessagesAsRead();

    // Set up a periodic reconnection checker
    _reconnectionTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      if (_hasConnectionError && !_isConnecting) {
        _retryConnection();
      }
    });

    _initChatConnection();

    // Add multiple post-frame callbacks to ensure scrolling works reliably
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollToBottom();

      // Add a second delayed scroll for reliability
      Future.delayed(Duration(milliseconds: 500), () {
        _scrollToBottom();
      });
    });
  }

  @override
  void dispose() {
    // Cancel all timers and subscriptions first
    _messageSubscription?.cancel();
    _reconnectionTimer?.cancel();
    _scrollController.dispose();
    _messageController.dispose();
    // Disconnect chat service last
    _chatService.disconnectFromChat();

    // Refresh unread count when leaving the chat
    _chatService.getUnreadMessageCount().then((count) {
      _chatService.unreadCount.listen((count) {
        debugPrint('📊 Updated unread count on dispose: $count');
      });
    });

    super.dispose();
  }

  Future<void> _getCurrentUser() async {
    final username = await SecureStorageService.getUsername();
    final userId = await SecureStorageService.getUserId();
    setState(() {
      _currentUsername = username;
      _currentUserId = userId;
    });
  }

  Future<void> _loadChatHistory() async {
    if (_isLoading) return;

    setState(() {
      _isLoading = true;
      _loadError = null;
    });

    try {
      debugPrint('📚 Loading chat history...');
      // Clean the friendId first to remove any widget hash symbols
      final friendId = ChatService.cleanId(widget.friendId);
      debugPrint('📚 Using cleaned friend ID: [$friendId]');

      // First remove any # character which can cause URL issues
      final sanitizedId = friendId.replaceAll('#', '');
      debugPrint('📚 Removed # characters: [$sanitizedId]');

      // Then get a numeric-only friend ID
      final numericFriendId = sanitizedId.replaceAll(RegExp(r'[^\d]'), '');

      if (numericFriendId.isEmpty) {
        throw Exception(
          'Invalid friend ID format - must contain at least one digit',
        );
      }

      debugPrint('📚 Using numeric friend ID: [$numericFriendId]');

      final chatHistory = await _chatService.getDirectChatHistory(
        numericFriendId,
      );

      if (!mounted) return;

      // Convert the map data to DirectMessage objects
      final directMessages = chatHistory.map((msg) {
        try {
          return DirectMessage.fromJson(msg);
        } catch (e) {
          debugPrint('❌ Error converting message: $e');
          // Create a fallback message with the data we have
          return DirectMessage(
            id: msg['id']?.toString() ?? '0',
            sender: msg['sender'] ?? '',
            receiver: msg['receiver'] ?? '',
            content: msg['content'] ?? 'Error loading message',
            timestamp:
                DateTime.tryParse(msg['timestamp'] ?? '') ?? DateTime.now(),
            isRead: msg['is_read'] ?? false,
            senderUsername: msg['sender_username'],
            receiverUsername: msg['receiver_username'],
          );
        }
      }).toList();

      setState(() {
        _messages = directMessages;
        _isLoading = false;
      });

      debugPrint('✅ Loaded ${_messages.length} messages');

      // Scroll to the bottom after loading history
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _scrollToBottom();
      });
    } catch (e) {
      if (!mounted) return;

      debugPrint('❌ Error loading chat history: $e');
      setState(() {
        _isLoading = false;
        _loadError = e.toString();
      });
    }
  }

  void _initChatConnection() {
    // Check if mounted before proceeding
    if (!mounted) return;

    setState(() {
      _isConnecting = true;
      _connectionError = null;
    });

    // Clean the friendId first to remove any widget hash symbols
    final friendId = ChatService.cleanId(widget.friendId);
    debugPrint('🔌 Initializing chat with friend ID: [$friendId]');

    try {
      // First remove any # character which can cause URL issues
      final sanitizedId = friendId.replaceAll('#', '');
      debugPrint('🔌 Removed # characters: [$sanitizedId]');

      // Then ensure friend ID is numeric only
      final numericFriendId = sanitizedId.replaceAll(RegExp(r'[^\d]'), '');

      if (numericFriendId.isEmpty) {
        throw Exception(
          'Invalid friend ID format - must contain at least one digit',
        );
      }

      debugPrint('🔌 Using numeric friend ID: [$numericFriendId]');

      // For debugging - show what the URL will likely be
      final String expectedUrl =
          '${ApiConfig.wsBaseUrl}/ws/chat/direct/$numericFriendId';
      debugPrint('🔌 Expected WebSocket URL: $expectedUrl');

      _chatService.connectToDirectChat(
        numericFriendId,
        onConnectionEstablished: () {
          if (!mounted) return;

          debugPrint('✅ WebSocket connection established');
          setState(() {
            _isConnecting = false;
            _connectionError = null;
            _hasConnectionError = false;
            _isConnected = true;
          });

          // Set up the message listener after connection
          _setupMessageListener();

          // Load chat history
          _loadChatHistory();
          _markMessagesAsRead();
        },
        onConnectionError: (error) {
          if (!mounted) return;

          debugPrint('❌ WebSocket connection error: $error');
          setState(() {
            _isConnecting = false;
            _connectionError = error;
            _hasConnectionError = true;
            _isConnected = false;
          });

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Connection error: $error'),
              backgroundColor: Colors.red,
              duration: Duration(seconds: 5),
              action: SnackBarAction(
                label: 'Retry',
                onPressed: _retryConnection,
                textColor: Colors.white,
              ),
            ),
          );
        },
      );
    } catch (e) {
      if (!mounted) return;

      debugPrint('❌ Error in _initChatConnection: $e');
      setState(() {
        _isConnecting = false;
        _connectionError = e.toString();
        _hasConnectionError = true;
        _isConnected = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error initializing chat: $e'),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 5),
          action: SnackBarAction(
            label: 'Retry',
            onPressed: _retryConnection,
            textColor: Colors.white,
          ),
        ),
      );
    }
  }

  void _setupMessageListener() {
    // Cancel any existing subscription first
    _messageSubscription?.cancel();

    // Listen for new messages
    _messageSubscription = _chatService.messagesStream.listen(
      (message) {
        // Check if widget is still mounted before proceeding
        if (!mounted) {
          _messageSubscription?.cancel();
          return;
        }

        debugPrint(
          '📱 UI received message: id=${message.id}, content="${message.content}"',
        );
        debugPrint(
          '📱 Message details: sender=${message.sender}, receiver=${message.receiver}',
        );

        // Handle duplicate detection more thoroughly
        bool isDuplicate = false;
        int localMessageIndex = -1;

        if (message.id != '0' && message.id.isNotEmpty) {
          // If we have a valid ID, check for duplicate by ID
          isDuplicate = _messages.any((m) => m.id == message.id);

          // Also check if we need to replace a temporary local message
          for (int i = 0; i < _messages.length; i++) {
            if (_messages[i].id.contains(
                      DateTime.now().year.toString(),
                    ) && // Likely a temp ID
                _messages[i].content == message.content &&
                _messages[i].sender == message.sender) {
              localMessageIndex = i;
              break;
            }
          }
        } else {
          // If no valid ID, check for duplicate by content, sender and time
          isDuplicate = _messages.any(
            (m) =>
                m.content == message.content &&
                m.sender == message.sender &&
                m.timestamp.difference(message.timestamp).inSeconds.abs() < 5,
          );
        }

        // Only proceed with setState if still mounted
        if (mounted) {
          setState(() {
            if (localMessageIndex >= 0) {
              debugPrint(
                '🔄 Replacing local message with server message at index $localMessageIndex',
              );
              _messages[localMessageIndex] = message;
            } else if (!isDuplicate) {
              debugPrint('✅ Adding new message to UI');
              _messages.add(message);
            } else {
              debugPrint('📱 Duplicate message detected, not adding to UI');
            }
          });

          // Scroll to bottom on new message if not a duplicate
          if (!isDuplicate || localMessageIndex >= 0) {
            _scrollToBottom();
          }

          // Mark as read if it's from the other user
          if (message.sender != _currentUserId) {
            _markMessagesAsRead();
          }
        }
      },
      onError: (error) {
        debugPrint('❌ Message stream error: $error');
        // Only show error if still mounted
        if (mounted) {
          _showErrorSnackBar('Message stream error: $error');
        }
      },
    );
  }

  void _scrollToBottom() {
    if (!mounted || !_scrollController.hasClients) return;

    try {
      _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
    } catch (e) {
      debugPrint('Initial scroll attempt failed: $e');
    }

    // Only schedule future operations if still mounted
    if (mounted) {
      Future.delayed(Duration(milliseconds: 100), () {
        if (!mounted || !_scrollController.hasClients) return;

        try {
          _scrollController.animateTo(
            _scrollController.position.maxScrollExtent,
            duration: Duration(milliseconds: 300),
            curve: Curves.easeOut,
          );
        } catch (e) {
          debugPrint('Animated scroll failed: $e');

          if (mounted) {
            Future.delayed(Duration(milliseconds: 500), () {
              if (!mounted || !_scrollController.hasClients) return;

              try {
                _scrollController.jumpTo(
                  _scrollController.position.maxScrollExtent,
                );
              } catch (e) {
                debugPrint('Final scroll attempt failed: $e');
              }
            });
          }
        }
      });
    }
  }

  Future<void> _markMessagesAsRead() async {
    if (!mounted) return;

    try {
      debugPrint('📱 Marking messages as read...');
      // Clean the friendId first to remove any widget hash symbols
      final friendId = ChatService.cleanId(widget.friendId);
      debugPrint('📱 Using cleaned friend ID: [$friendId]');

      // First remove any # character which can cause URL issues
      final sanitizedId = friendId.replaceAll('#', '');
      debugPrint('📱 Removed # characters: [$sanitizedId]');

      // Then get numeric-only friend ID
      final numericFriendId = sanitizedId.replaceAll(RegExp(r'[^\d]'), '');

      if (numericFriendId.isEmpty) {
        throw Exception(
            'Invalid friend ID format - must contain at least one digit');
      }

      debugPrint('📱 Using numeric friend ID: [$numericFriendId]');

      // First get or create the conversation
      final conversation =
          await _chatService.createOrGetConversation(numericFriendId);
      debugPrint('📝 Got conversation ID: ${conversation.id}');

      // Mark the conversation as read
      await _chatService.markConversationAsRead(conversation.id);
      debugPrint('✅ Marked conversation as read');

      // Update local message states
      setState(() {
        for (var message in _messages) {
          if (message.sender == numericFriendId) {
            message.isRead = true;
          }
        }
      });

      // Update unread count
      final unreadCount = await _chatService.getUnreadMessageCount();
      debugPrint('📊 New unread count: $unreadCount');
      _chatService.unreadCount.listen((count) {
        debugPrint('📊 Updated unread count in stream: $count');
      });
    } catch (e) {
      debugPrint('❌ Error marking messages as read: $e');
    }
  }

  void _sendMessage() {
    if (!mounted) return;

    final message = _messageController.text.trim();
    if (message.isEmpty) return;

    // Clean the friendId first to remove any widget hash symbols
    final friendId = ChatService.cleanId(widget.friendId);
    debugPrint('📤 Using cleaned friend ID: [$friendId]');

    // First remove any # character which can cause URL issues
    final sanitizedId = friendId.replaceAll('#', '');
    debugPrint('📤 Removed # characters: [$sanitizedId]');

    // Then get numeric-only friend ID
    final String numericFriendId = sanitizedId.replaceAll(RegExp(r'[^\d]'), '');

    if (numericFriendId.isEmpty) {
      _showErrorSnackBar('Invalid friend ID: must contain at least one digit');
      return;
    }

    debugPrint('📤 Sending message: "$message" to friend: [$numericFriendId]');

    setState(() {
      _isSending = true;
    });

    // Add local message immediately for better UX
    final localMessage = DirectMessage(
      id: '${DateTime.now().millisecondsSinceEpoch}', // Temporary local ID
      sender: _currentUserId ?? '',
      receiver: numericFriendId,
      content: message,
      timestamp: DateTime.now(),
      isRead: false,
      senderUsername: _currentUsername ?? 'Me',
      receiverUsername: widget.friendName,
    );

    setState(() {
      _messages.add(localMessage);
    });

    // Scroll to show the new message
    _scrollToBottom();

    // Send the message
    _chatService
        .sendDirectMessage(
      receiverId: numericFriendId,
      senderUsername: _currentUsername ?? 'Me',
      receiverUsername: widget.friendName,
      content: message,
    )
        .then((success) {
      if (!mounted) return;
      setState(() {
        _isSending = false;
        if (success) {
          debugPrint('✅ Message sent successfully');
          _messageController.clear();
        } else {
          debugPrint('❌ Failed to send message');
          _showErrorSnackBar('Failed to send message. Please try again.');

          // Remove the local message if sending failed
          setState(() {
            _messages.removeWhere(
              (msg) =>
                  msg.id == localMessage.id &&
                  msg.content == localMessage.content,
            );
          });
        }
      });
    }).catchError((error) {
      if (!mounted) return;
      debugPrint('❌ Error sending message: $error');
      setState(() {
        _isSending = false;
        // Remove the local message on error
        _messages.removeWhere(
          (msg) =>
              msg.id == localMessage.id && msg.content == localMessage.content,
        );
      });
      _showErrorSnackBar('Error sending message: $error');
    });
  }

  void _showErrorSnackBar(String message) {
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 3),
        action: SnackBarAction(
          label: 'Dismiss',
          textColor: Colors.white,
          onPressed: () {
            ScaffoldMessenger.of(context).hideCurrentSnackBar();
          },
        ),
      ),
    );
  }

  Widget _buildConnectionErrorBanner() {
    if (!_hasConnectionError) return const SizedBox.shrink();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      color: _isConnecting ? Colors.amber.shade800 : Colors.red.shade800,
      child: Row(
        children: [
          Icon(
            _isConnecting ? Icons.sync : Icons.error_outline,
            color: Colors.white,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _connectionError ?? 'Connection error',
              style: const TextStyle(color: Colors.white),
            ),
          ),
          TextButton(
            onPressed: _isConnecting ? null : _retryConnection,
            style: TextButton.styleFrom(
              padding: EdgeInsets.symmetric(horizontal: 8, vertical: 0),
              foregroundColor: Colors.white,
            ),
            child: _isConnecting
                ? Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'CONNECTING...',
                        style: TextStyle(fontSize: 12),
                      ),
                    ],
                  )
                : const Text(
                    'RETRY',
                    style: TextStyle(fontWeight: FontWeight.bold),
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
          Icon(Icons.chat_bubble_outline, size: 64, color: Colors.grey),
          SizedBox(height: 16),
          Text(
            'No messages yet',
            style: TextStyle(
              fontSize: 18,
              color: Colors.grey,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 8),
          Text(
            'Start the conversation!',
            style: TextStyle(fontSize: 14, color: Colors.grey),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.friendName),
        elevation: 1,
        actions: [
          if (_isConnected || !_isConnecting)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _loadChatHistory,
              tooltip: 'Refresh messages',
            ),
        ],
      ),
      body: Column(
        children: [
          // Error banners - either connection error or load error
          if (_hasConnectionError) _buildConnectionErrorBanner(),
          if (_loadError != null && !_hasConnectionError)
            _buildErrorBanner(_loadError!),

          // Main content
          Expanded(
            child: Stack(
              children: [
                // Messages list
                if (_messages.isEmpty &&
                    !_isLoading &&
                    !_hasConnectionError &&
                    _loadError == null)
                  _buildEmptyState()
                else if (!_isLoading && !_hasConnectionError)
                  _buildMessageList(),

                // Loading indicator
                if (_isLoading)
                  const Center(child: CircularProgressIndicator()),
              ],
            ),
          ),

          // Message input field - only show when not loading or having errors
          if (!_hasConnectionError && !_isLoading && _loadError == null)
            _buildMessageInput(),
        ],
      ),
    );
  }

  Widget _buildErrorBanner(String error) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      color: Colors.amber,
      child: Row(
        children: [
          const Icon(Icons.warning, color: Colors.black87),
          const SizedBox(width: 12),
          Expanded(
            child: Text(error, style: const TextStyle(color: Colors.black87)),
          ),
          TextButton(
            onPressed: _loadChatHistory,
            child: const Text(
              'RETRY',
              style: TextStyle(
                color: Colors.black87,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageList() {
    return ListView.builder(
      controller: _scrollController,
      padding: EdgeInsets.all(16),
      itemCount: _messages.length,
      itemBuilder: (context, index) {
        final message = _messages[index];
        final isMe = message.sender == _currentUserId;

        return MessageBubble(
          message: message,
          isMe: isMe,
          isDarkMode: widget.isDarkMode,
        );
      },
    );
  }

  Widget _buildMessageInput() {
    final backgroundColor =
        widget.isDarkMode ? Colors.grey[900] : Colors.grey[100];
    final textColor = widget.isDarkMode ? Colors.white : Colors.black;

    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      decoration: BoxDecoration(
        color: backgroundColor,
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 4,
            offset: Offset(0, -2),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _messageController,
                style: TextStyle(color: textColor),
                decoration: InputDecoration(
                  hintText: 'Type a message...',
                  hintStyle: TextStyle(color: Colors.grey),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                    borderSide: BorderSide.none,
                  ),
                  filled: true,
                  fillColor: backgroundColor,
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),
                ),
                maxLines: null,
                textCapitalization: TextCapitalization.sentences,
                onSubmitted: (_) => _sendMessage(),
              ),
            ),
            SizedBox(width: 8),
            Material(
              color: Theme.of(context).primaryColor,
              borderRadius: BorderRadius.circular(24),
              child: InkWell(
                borderRadius: BorderRadius.circular(24),
                onTap: _sendMessage,
                child: Container(
                  padding: EdgeInsets.all(10),
                  child: Icon(Icons.send, color: Colors.white),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _retryConnection() {
    debugPrint('🔄 Retrying connection to chat...');
    _initChatConnection();
  }
}
