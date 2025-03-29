import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/io.dart';
import 'package:echo_journal_1/services/auth/secure_storage_service.dart';
import 'package:echo_journal_1/models/chat/conversation.dart';
import 'package:echo_journal_1/models/chat/message.dart';
import 'package:echo_journal_1/services/api/api_service.dart';
import 'package:echo_journal_1/data/models/direct_message.dart';
import 'package:echo_journal_1/core/configs/api_config.dart';

class ChatService {
  // Static helper to clean any IDs that might come from widgets with hash codes
  static String cleanId(String id) {
    // If the ID contains a hash symbol, take only the part before it
    if (id.contains('#')) {
      final parts = id.split('#');
      return parts[0];
    }
    return id;
  }

  final ApiService _apiService = ApiService();
  WebSocketChannel? _channel;
  final _messagesController = StreamController<List<Message>>.broadcast();
  final _conversationsController =
      StreamController<List<Conversation>>.broadcast();
  final _unreadCountController = StreamController<int>.broadcast();
  final _directMessagesController = StreamController<DirectMessage>.broadcast();
  StreamSubscription? _messageSubscription;
  String? _currentUserId;
  List<Message> _messages = [];

  // Ping timer to keep the connection alive
  Timer? _pingTimer;

  ChatService() {
    _initCurrentUser();
  }

  Future<void> _initCurrentUser() async {
    _currentUserId = await SecureStorageService.getUserId();
  }

  Stream<List<Message>> get messages => _messagesController.stream;
  Stream<List<Conversation>> get conversations =>
      _conversationsController.stream;
  Stream<int> get unreadCount => _unreadCountController.stream;
  Stream<DirectMessage> get messagesStream => _directMessagesController.stream;

  Future<String?> _getToken() async {
    return await SecureStorageService.getAccessToken();
  }

  Future<List<Conversation>> getRecentConversations() async {
    try {
      debugPrint('⚡ Getting recent conversations...');
      final response = await _apiService.get(
        ApiConfig.getFullUrl(ApiConfig.recentConversationsEndpoint),
      );
      if (response.statusCode == 200) {
        debugPrint('✅ Got response for recent conversations');
        debugPrint('Raw response data type: ${response.data.runtimeType}');
        debugPrint('Raw response data: ${response.data}');

        // Handle different response formats
        List<dynamic> conversationsData = [];
        try {
          if (response.data is Map &&
              response.data.containsKey('conversations')) {
            debugPrint('📦 Response is a Map with conversations key');
            conversationsData = response.data['conversations'] as List<dynamic>;
          } else if (response.data is List) {
            debugPrint('📦 Response is a List');
            conversationsData = response.data as List<dynamic>;
          } else {
            debugPrint(
              '❌ Unexpected response format: ${response.data.runtimeType}',
            );
            debugPrint(
              'Response data keys: ${response.data is Map ? (response.data as Map).keys.toList() : "not a map"}',
            );
          }
        } catch (e) {
          debugPrint('❌ Error extracting conversation data: $e');
        }

        debugPrint('📋 Processing ${conversationsData.length} conversations');
        final conversations = conversationsData.map((json) {
          try {
            debugPrint('🔍 Processing conversation: ${json['id']}');
            // Convert IDs to strings and ensure required fields exist
            final processedJson = Map<String, dynamic>.from(json);

            // Debug the type of each field
            processedJson.forEach((key, value) {
              debugPrint('  $key: ${value.runtimeType} = $value');
            });

            processedJson['id'] = processedJson['id']?.toString() ?? '';
            processedJson['created_at'] =
                processedJson['created_at']?.toString() ??
                    DateTime.now().toIso8601String();
            processedJson['updated_at'] =
                processedJson['updated_at']?.toString() ??
                    DateTime.now().toIso8601String();
            processedJson['unread_count'] = processedJson['unread_count'] ?? 0;

            // Ensure participants is a list
            if (processedJson['participants'] == null) {
              processedJson['participants'] = [];
            } else if (processedJson['participants'] is List) {
              // Process each participant to ensure ID is string
              final participantsList = processedJson['participants'] as List;
              for (int i = 0; i < participantsList.length; i++) {
                final participant = participantsList[i];
                if (participant is Map<String, dynamic>) {
                  participant['id'] = participant['id']?.toString() ?? '';
                }
              }
            }

            final conversation = Conversation.fromJson(processedJson);
            debugPrint(
              '✅ Successfully parsed conversation ${conversation.id}',
            );
            return conversation;
          } catch (e) {
            debugPrint('❌ Error parsing conversation: $e');
            debugPrint('JSON data: $json');

            // Create a minimal valid conversation instead of throwing
            final Map<String, dynamic> fallbackJson = {
              'id': '0',
              'participants': [],
              'created_at': DateTime.now().toIso8601String(),
              'updated_at': DateTime.now().toIso8601String(),
              'unread_count': 0,
            };
            return Conversation.fromJson(fallbackJson);
          }
        }).toList();

        debugPrint(
          '✅ Parsed ${conversations.length} conversations successfully',
        );
        _conversationsController.add(conversations);
        return conversations;
      } else {
        throw Exception('Failed to load conversations: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('❌ Error getting recent conversations: $e');
      _conversationsController.addError(e);
      throw Exception('Failed to load conversations: $e');
    }
  }

  Future<List<Message>> getChatHistory(String conversationId) async {
    try {
      final response = await _apiService.get(
        ApiConfig.getFullUrl(
          '${ApiConfig.conversationsEndpoint}/$conversationId/messages/',
        ),
      );
      if (response.statusCode == 200) {
        final List<dynamic> data = response.data;
        final messages = data.map((json) => Message.fromJson(json)).toList();
        _messagesController.add(messages);
        return messages;
      } else {
        throw Exception('Failed to load chat history: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error getting chat history: $e');
      _messagesController.addError(e);
      throw Exception('Failed to load chat history: $e');
    }
  }

  Future<Conversation> createOrGetConversation(String userId) async {
    try {
      final response = await _apiService.post(
        ApiConfig.getFullUrl(ApiConfig.conversationsEndpoint),
        data: {'user_id': userId},
      );
      if (response.statusCode == 201 || response.statusCode == 200) {
        final conversation = Conversation.fromJson(response.data);
        // Update conversations list
        final currentConversations = await getRecentConversations();
        if (!currentConversations.any((c) => c.id == conversation.id)) {
          currentConversations.insert(0, conversation);
          _conversationsController.add(currentConversations);
        }
        return conversation;
      } else {
        throw Exception(
          'Failed to create/get conversation: ${response.statusCode}',
        );
      }
    } catch (e) {
      debugPrint('Error creating/getting conversation: $e');
      throw Exception('Failed to create/get conversation: $e');
    }
  }

  Future<void> sendMessage(String conversationId, String content) async {
    try {
      final response = await _apiService.post(
        ApiConfig.getFullUrl(
          '${ApiConfig.conversationsEndpoint}/$conversationId/send_message/',
        ),
        data: {'content': content},
      );
      if (response.statusCode == 201) {
        final message = Message.fromJson(response.data);
        // Update messages list
        final currentMessages = await getChatHistory(conversationId);
        currentMessages.add(message);
        _messagesController.add(currentMessages);

        // Update conversations list to show latest message
        await getRecentConversations();
      } else {
        throw Exception('Failed to send message: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error sending message: $e');
      throw Exception('Failed to send message: $e');
    }
  }

  Future<int> getUnreadMessageCount() async {
    try {
      debugPrint('⚡ Getting unread message count...');
      final response = await _apiService.get(
        ApiConfig.getFullUrl(ApiConfig.unreadCountEndpoint),
      );

      if (response.statusCode == 200) {
        debugPrint('✅ Got unread count response: ${response.data}');

        // Handle different response formats
        if (response.data == null) return 0;

        if (response.data is int) {
          return response.data;
        }

        if (response.data is Map) {
          final unreadCount = response.data['unread_count'];
          if (unreadCount != null) {
            if (unreadCount is int) return unreadCount;
            if (unreadCount is String) return int.tryParse(unreadCount) ?? 0;
          }

          // Try to find any field that might contain the count
          for (var entry in response.data.entries) {
            if (entry.value is int) return entry.value;
            if (entry.value is String) {
              final parsed = int.tryParse(entry.value);
              if (parsed != null) return parsed;
            }
          }
        }

        if (response.data is String) {
          return int.tryParse(response.data) ?? 0;
        }

        debugPrint('⚠️ Could not parse unread count from response');
        return 0;
      }

      debugPrint('❌ Failed to get unread count: ${response.statusCode}');
      return 0;
    } catch (e) {
      debugPrint('❌ Error getting unread count: $e');
      return 0;
    }
  }

  Future<void> markConversationAsRead(String conversationId) async {
    try {
      debugPrint('⚡ Marking conversation $conversationId as read');
      final response = await _apiService.post(
        ApiConfig.getFullUrl(
          '${ApiConfig.conversationsEndpoint}/$conversationId/mark_read',
        ),
      );
      if (response.statusCode == 200) {
        debugPrint('✅ Successfully marked conversation as read');

        // Update messages to mark them as read
        final messages = await getChatHistory(conversationId);
        messages.forEach((m) => m.isRead = true);
        _messagesController.add(messages);

        // Update conversations list and unread count
        await getRecentConversations();
        final unreadCount = await getUnreadMessageCount();
        _unreadCountController.add(unreadCount);

        debugPrint('✅ Updated unread count: $unreadCount');
      } else {
        debugPrint(
            '❌ Failed to mark conversation as read: ${response.statusCode}');
        throw Exception(
          'Failed to mark conversation as read: ${response.statusCode}',
        );
      }
    } catch (e) {
      debugPrint('❌ Error marking conversation as read: $e');
      throw Exception('Failed to mark conversation as read: $e');
    }
  }

  Future<void> connectToDirectChat(
    String friendId, {
    Function? onConnectionEstablished,
    Function(String)? onConnectionError,
  }) async {
    // Close any existing connection first
    await disconnectFromChat(isDeliberate: false);

    _isDeliberateDisconnect = false;

    try {
      // Reset deliberate disconnect flag
      _lastConnectedFriendId = friendId;

      debugPrint('⚡ Connecting to direct chat with friend ID: $friendId');

      // Ensure the friend ID is numeric only using helper method
      final sanitizedFriendId = _ensureNumericFriendId(friendId);
      debugPrint('🔑 Using numeric friend ID: $sanitizedFriendId');

      final accessToken = await _getToken();
      if (accessToken == null) throw Exception('User is not authenticated');

      // Construct a simple, clean WebSocket URL
      final String wsUrlString =
          '${ApiConfig.wsBaseUrl}/ws/chat/direct/$sanitizedFriendId';
      debugPrint('🔌 WebSocket URL: $wsUrlString');

      // Final safety check - make sure URL doesn't contain any invalid characters
      if (wsUrlString.contains('#') ||
          wsUrlString.contains('?') ||
          wsUrlString.contains('&') ||
          sanitizedFriendId.contains(RegExp(r'[^\d]'))) {
        throw Exception(
          'WebSocket URL contains invalid characters: $wsUrlString',
        );
      }

      debugPrint(
        '🔌 WebSocket Headers: Authorization: Bearer ${accessToken.substring(0, 10)}...',
      );

      try {
        // Create WebSocket connection
        _channel = IOWebSocketChannel.connect(
          wsUrlString,
          pingInterval: const Duration(seconds: 30),
          headers: {'Authorization': 'Bearer $accessToken'},
        );
        debugPrint('🔌 WebSocket channel created');
      } catch (e) {
        debugPrint('❌ Error creating WebSocket channel: $e');
        throw Exception('Failed to create WebSocket channel: $e');
      }

      // Listen for connection established message with timeout
      bool connectionEstablished = false;
      final connectionTimeout = Timer(Duration(seconds: 5), () {
        if (!connectionEstablished) {
          debugPrint('❌ WebSocket connection timed out');
          if (_channel != null) {
            try {
              _channel?.sink.close();
              _channel = null;
            } catch (e) {
              debugPrint('❌ Error closing channel: $e');
            }
          }
          throw Exception('Connection timed out');
        }
      });

      // Set up initial connection listener
      StreamSubscription? initialListener;
      initialListener = _channel?.stream.listen(
        (message) {
          try {
            debugPrint('📥 Initial connection message: $message');
            final data = jsonDecode(message);

            // Check for connection established message
            if (data is Map &&
                data.containsKey('type') &&
                data['type'] == 'connection_established') {
              connectionEstablished = true;
              connectionTimeout.cancel();
              initialListener?.cancel();

              debugPrint('✅ WebSocket connection established');

              // Set up regular message listener
              _setupMessageListener();

              // Start ping timer
              _startPingTimer();

              // Notify success
              if (onConnectionEstablished != null) {
                onConnectionEstablished();
              }
            }
          } catch (e) {
            debugPrint('❌ Error in initial connection: $e');
          }
        },
        onError: (error) {
          debugPrint('❌ WebSocket connection error: $error');
          connectionTimeout.cancel();

          String errorMessage = _formatConnectionError(error);
          if (onConnectionError != null) {
            onConnectionError(errorMessage);
          }
        },
        onDone: () {
          // Only consider this an error if we haven't established connection yet
          if (!connectionEstablished) {
            debugPrint('❌ WebSocket connection closed before established');
            connectionTimeout.cancel();

            if (onConnectionError != null) {
              onConnectionError('Connection closed unexpectedly');
            }
          }
        },
      );

      // Send a test ping to verify connection
      _channel?.sink.add(jsonEncode({'type': 'ping'}));
    } catch (e) {
      debugPrint('❌ Error connecting to direct chat: $e');

      // Clean up any lingering resources
      if (_channel != null) {
        try {
          _channel?.sink.close();
          _channel = null;
        } catch (_) {}
      }

      // Format the error message
      String errorMessage = _formatConnectionError(e);

      if (onConnectionError != null) {
        onConnectionError(errorMessage);
      }

      _directMessagesController.addError(Exception(errorMessage));
      throw Exception('Failed to connect: $errorMessage');
    }
  }

  // Helper to check if a string is numeric
  bool _isNumeric(String? str) {
    if (str == null) return false;
    return RegExp(r'^\d+$').hasMatch(str);
  }

  // Helper to format connection errors in a user-friendly way
  String _formatConnectionError(dynamic error) {
    String errorStr = error.toString();

    if (errorStr.contains('404')) {
      return 'Chat server not available (404)';
    } else if (errorStr.contains('403')) {
      return 'Access denied (403)';
    } else if (errorStr.contains('401')) {
      return 'Authentication required (401)';
    } else if (errorStr.contains('timeout')) {
      return 'Connection timed out';
    } else if (errorStr.contains('refused')) {
      return 'Connection refused';
    } else if (errorStr.contains('network is unreachable')) {
      return 'Network unreachable';
    } else if (errorStr.contains('SocketException')) {
      return 'Network error - server may be down';
    } else if (errorStr.contains('WebSocketException')) {
      return 'WebSocket connection error';
    }

    return 'Connection error: $errorStr';
  }

  // Add a reconnect method with exponential backoff
  Future<void> reconnectToDirectChat(
    String friendId, {
    int attempt = 1,
    Function? onReconnected,
    Function(String)? onReconnectFailed,
  }) async {
    if (attempt > 5) {
      debugPrint('❌ Maximum reconnection attempts reached');
      final errorMsg = 'Could not reconnect after multiple attempts';
      _directMessagesController.addError(Exception(errorMsg));

      if (onReconnectFailed != null) {
        onReconnectFailed(errorMsg);
      }
      return;
    }

    try {
      debugPrint('🔄 Attempting to reconnect (attempt $attempt)...');

      // Ensure friendId is numeric using helper method
      final numericFriendId = _ensureNumericFriendId(friendId);
      debugPrint(
        '🔑 Using numeric friend ID for reconnection: $numericFriendId',
      );

      await connectToDirectChat(
        numericFriendId,
        onConnectionEstablished: onReconnected,
        onConnectionError: (error) {
          debugPrint('❌ Reconnection attempt $attempt failed: $error');

          // Exponential backoff: wait longer between each attempt
          final waitTime = Duration(seconds: attempt * 2);
          debugPrint(
            '⏱️ Waiting ${waitTime.inSeconds} seconds before next attempt',
          );

          Future.delayed(waitTime, () {
            reconnectToDirectChat(
              numericFriendId,
              attempt: attempt + 1,
              onReconnected: onReconnected,
              onReconnectFailed: onReconnectFailed,
            );
          });

          if (onReconnectFailed != null) {
            onReconnectFailed(
              'Reconnection failed (attempt $attempt). Retrying in ${waitTime.inSeconds} seconds...',
            );
          }
        },
      );

      debugPrint('✅ Reconnected successfully');
    } catch (e) {
      debugPrint('❌ Reconnection attempt $attempt failed: $e');

      if (attempt < 5) {
        // Exponential backoff: wait longer between each attempt
        final waitTime = Duration(seconds: attempt * 2);
        debugPrint(
          '⏱️ Waiting ${waitTime.inSeconds} seconds before next attempt',
        );

        Future.delayed(waitTime, () {
          reconnectToDirectChat(
            friendId,
            attempt: attempt + 1,
            onReconnected: onReconnected,
            onReconnectFailed: onReconnectFailed,
          );
        });

        if (onReconnectFailed != null) {
          onReconnectFailed(
            'Reconnection failed (attempt $attempt). Retrying in ${waitTime.inSeconds} seconds...',
          );
        }
      } else {
        final errorMsg = 'Could not reconnect after multiple attempts';
        if (onReconnectFailed != null) {
          onReconnectFailed(errorMsg);
        }
      }
    }
  }

  void _setupMessageListener() {
    // Cancel existing subscription if any
    _messageSubscription?.cancel();

    _messageSubscription = _channel?.stream.listen(
      (message) {
        try {
          debugPrint('📥 Raw WebSocket message received: $message');

          final data = jsonDecode(message);
          debugPrint('📥 Decoded message: $data');

          // Exit early if not a map
          if (data is! Map) {
            debugPrint('⚠️ Received message is not a map');
            return;
          }

          // Handle different message types
          if (data.containsKey('type')) {
            final messageType = data['type'];

            // Handle ping messages
            if (messageType == 'ping') {
              debugPrint('📍 Ping received, sending pong');
              _channel?.sink.add(jsonEncode({'type': 'pong'}));
              return;
            }

            // Handle pong responses (from our pings)
            if (messageType == 'pong') {
              debugPrint('📍 Pong received from server');
              return;
            }

            // Handle connection confirmation - already handled in the connect method
            if (messageType == 'connection_established') {
              debugPrint('📍 Connection confirmation received');
              return;
            }

            // Handle error messages
            if (messageType == 'error') {
              debugPrint('❌ Server error: ${data['error']}');
              _directMessagesController.addError(Exception(data['error']));
              return;
            }
          }

          // If we reach here, this is a regular chat message
          debugPrint('🔄 Processing chat message: $data');

          // Ensure all required fields for a chat message exist
          if (!_isValidChatMessage(data)) {
            debugPrint('⚠️ Received invalid chat message format: $data');
            return;
          }

          // Convert to correct type for fromJson
          final stringKeyedData = Map<String, dynamic>.from(data);

          // Process the received message and broadcast it
          final directMessage = DirectMessage.fromJson(stringKeyedData);
          debugPrint('✅ Message processed: ${directMessage.content}');
          _directMessagesController.add(directMessage);

          // If the message is from someone else, mark it as read
          if (directMessage.sender != _currentUserId) {
            debugPrint('📬 Message is from someone else, marking as read');
            // Get the conversation ID from the message or create a new conversation
            createOrGetConversation(directMessage.sender).then((conversation) {
              debugPrint('📝 Got conversation ID: ${conversation.id}');
              markConversationAsRead(conversation.id).then((_) {
                debugPrint('✅ Marked conversation as read');
              }).catchError((e) {
                debugPrint('❌ Error marking conversation as read: $e');
              });
            }).catchError((e) {
              debugPrint('❌ Error getting conversation: $e');
            });
          }
        } catch (e) {
          debugPrint('❌ Error processing message: $e');
          debugPrint('Raw message: $message');
          // Don't propagate parsing errors to the stream
        }
      },
      onError: (error) {
        debugPrint('❌ WebSocket error: $error');
        _directMessagesController.addError(error);
        _handleConnectionError(error);
      },
      onDone: () {
        debugPrint('⚠️ WebSocket connection closed');
        _channel = null;
        _handleConnectionClosed();
      },
    );
  }

  // Helper to validate a chat message has all required fields
  bool _isValidChatMessage(Map<dynamic, dynamic> data) {
    final requiredFields = [
      'message',
      'content',
      'sender',
      'sender_id',
      'receiver',
      'receiver_id',
      'timestamp',
    ];

    for (final field in requiredFields) {
      if (!data.containsKey(field) || data[field] == null) {
        debugPrint('⚠️ Missing required field: $field');
        return false;
      }
    }

    return true;
  }

  void _handleConnectionError(dynamic error) {
    // Don't auto-reconnect if we're deliberately disconnecting
    if (_isDeliberateDisconnect) return;

    String errorMessage = _formatConnectionError(error);
    debugPrint('🔄 Connection error: $errorMessage, will attempt to reconnect');

    // Wait a bit before reconnecting
    Future.delayed(const Duration(seconds: 2), () {
      // We'll handle reconnection in the _handleConnectionClosed method
    });
  }

  bool _isDeliberateDisconnect = false;
  String? _lastConnectedFriendId;

  // Helper to ensure friend ID is numeric only
  String _ensureNumericFriendId(String friendId) {
    // First clean any hash symbols
    final cleanedId = cleanId(friendId);

    // Then remove any non-numeric characters
    final numericId = cleanedId.replaceAll(RegExp(r'[^\d]'), '');

    if (numericId.isEmpty) {
      throw Exception('Invalid friend ID: must contain at least one digit');
    }

    return numericId;
  }

  // Helper method to process message data
  Map<String, dynamic> _processMessageData(
    Map<String, dynamic> messageData,
    String friendId,
  ) {
    final processedMessage = <String, dynamic>{};

    // Handle message content (support both 'content' and 'message' fields)
    processedMessage['content'] = messageData['content'] ??
        messageData['message'] ??
        'Message content unavailable';

    // Handle timestamp (support both 'timestamp' and 'created_at' fields)
    final timestamp = messageData['timestamp'] ??
        messageData['created_at'] ??
        DateTime.now().toIso8601String();
    processedMessage['timestamp'] = timestamp;

    // Handle sender information
    processedMessage['sender'] = messageData['sender'] ??
        messageData['sender_id']?.toString() ??
        friendId;

    // Handle receiver information
    processedMessage['receiver'] = messageData['receiver'] ??
        messageData['receiver_id']?.toString() ??
        'current_user';

    // Add any additional fields that might be useful
    processedMessage['message_id'] = messageData['id'] ??
        messageData['message_id'] ??
        DateTime.now().millisecondsSinceEpoch.toString();

    return processedMessage;
  }

  // Helper method to create a fallback message
  Map<String, dynamic> _createFallbackMessage(String friendId) {
    return {
      'content': 'Message unavailable',
      'timestamp': DateTime.now().toIso8601String(),
      'sender': friendId,
      'receiver': 'current_user',
      'message_id': 'fallback_${DateTime.now().millisecondsSinceEpoch}',
    };
  }

  void _handleConnectionClosed() {
    // Don't auto-reconnect if we're deliberately disconnecting
    if (_isDeliberateDisconnect) {
      debugPrint('🛑 Deliberate disconnect - not attempting to reconnect');
      return;
    }

    debugPrint('🔄 Connection closed unexpectedly, will attempt to reconnect');

    // Only reconnect if we have a friend ID
    if (_lastConnectedFriendId != null) {
      // Don't attempt reconnect if channel is already being established
      if (_channel != null) {
        debugPrint('⚠️ Channel already exists, not reconnecting');
        return;
      }

      Future.delayed(const Duration(seconds: 2), () {
        if (!_isDeliberateDisconnect) {
          try {
            // Ensure friend ID is numeric only
            final numericFriendId = _ensureNumericFriendId(
              _lastConnectedFriendId!,
            );

            debugPrint(
              '🔄 Attempting to reconnect to friend: $numericFriendId',
            );
            reconnectToDirectChat(numericFriendId);
          } catch (e) {
            debugPrint('❌ Cannot reconnect: $e');
          }
        }
      });
    } else {
      debugPrint('⚠️ No friend ID available for reconnection');
    }
  }

  Future<void> disconnectFromChat({bool isDeliberate = true}) async {
    try {
      // Set the deliberate disconnect flag
      _isDeliberateDisconnect = isDeliberate;

      // Cancel ping timer
      _pingTimer?.cancel();
      _pingTimer = null;

      _messageSubscription?.cancel();
      await _channel?.sink.close();
      _channel = null;
      debugPrint('Disconnected from chat (deliberate: $isDeliberate)');

      if (isDeliberate) {
        _lastConnectedFriendId = null;
      }
    } catch (e) {
      debugPrint('Error disconnecting from chat: $e');
    }
  }

  void dispose() {
    try {
      // Set deliberate disconnect flag to prevent reconnection attempts
      _isDeliberateDisconnect = true;

      // Cancel ping timer
      _pingTimer?.cancel();
      _pingTimer = null;

      // Cancel message subscription
      _messageSubscription?.cancel();
      _messageSubscription = null;

      // Close channel
      _channel?.sink.close();
      _channel = null;

      // Close all stream controllers
      _messagesController.close();
      _conversationsController.close();
      _unreadCountController.close();
      _directMessagesController.close();

      debugPrint('✅ ChatService disposed and all resources cleaned up');
    } catch (e) {
      debugPrint('❌ Error during ChatService disposal: $e');
    }
  }

  Future<List<Map<String, dynamic>>> getDirectChatHistory(
    String friendId,
  ) async {
    try {
      debugPrint('⚡ Getting direct chat history for friend ID: $friendId');

      // Clean and validate the friend ID
      final numericFriendId = _ensureNumericFriendId(friendId);
      debugPrint('🔍 Using numeric friend ID: $numericFriendId');

      // Get current user ID
      final currentUserId = await SecureStorageService.getUserId();
      if (currentUserId == null) {
        throw Exception('User not authenticated');
      }

      // Construct the URL for chat history
      final url =
          '${ApiConfig.messagesEndpoint}/with_user?user_id=$numericFriendId';
      debugPrint('🔗 Request URL: ${ApiConfig.getFullUrl(url)}');

      final response = await _apiService.get(ApiConfig.getFullUrl(url));

      debugPrint('📥 Response status code: ${response.statusCode}');
      debugPrint('📥 Raw response data: ${response.data}');
      debugPrint('📥 Response data type: ${response.data.runtimeType}');

      if (response.statusCode == 200) {
        // Try to handle any possible response format
        List<dynamic> messagesData = [];

        try {
          if (response.data == null) {
            debugPrint('⚠️ Response data is null');
            return [];
          }

          if (response.data is List) {
            debugPrint('📦 Response is a List');
            messagesData = response.data;
          } else if (response.data is Map) {
            debugPrint('📦 Response is a Map');
            if (response.data.containsKey('messages')) {
              messagesData = response.data['messages'] as List<dynamic>;
            } else if (response.data.containsKey('data')) {
              messagesData = response.data['data'] as List<dynamic>;
            } else if (response.data.containsKey('history')) {
              messagesData = response.data['history'] as List<dynamic>;
            } else {
              // If it's a single message, wrap it in a list
              messagesData = [response.data];
            }
          }
        } catch (e) {
          debugPrint('❌ Error parsing response data: $e');
          debugPrint('Raw response data: ${response.data}');
          return [];
        }

        debugPrint('📋 Processing ${messagesData.length} messages');

        // Filter messages to ensure they're between current user and friend
        final List<Map<String, dynamic>> validMessages = [];
        for (final json in messagesData) {
          try {
            // Convert to Map<String, dynamic> if it's not already
            final Map<String, dynamic> messageData;
            if (json is Map<String, dynamic>) {
              messageData = json;
            } else if (json is Map) {
              messageData = Map<String, dynamic>.from(json);
            } else {
              debugPrint('⚠️ Invalid message format: $json');
              continue;
            }

            // Verify message is between current user and friend
            final sender = messageData['sender']?.toString() ?? '';
            final receiver = messageData['receiver']?.toString() ?? '';
            if ((sender == currentUserId && receiver == numericFriendId) ||
                (sender == numericFriendId && receiver == currentUserId)) {
              validMessages.add(messageData);
            } else {
              debugPrint(
                  '⚠️ Skipping message not in this conversation: $messageData');
            }
          } catch (e) {
            debugPrint('❌ Error processing message: $e');
            debugPrint('Message data: $json');
            continue;
          }
        }
        return validMessages;
      } else {
        debugPrint('❌ Failed to load chat history: ${response.statusCode}');
        debugPrint('Error response: ${response.data}');
        throw Exception('Failed to load chat history: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('❌ Error getting direct chat history: $e');
      throw Exception('Failed to load chat history: $e');
    }
  }

  Future<bool> markMessagesAsRead(String friendId) async {
    try {
      debugPrint('⚡ Marking messages as read for friend ID: $friendId');

      // Ensure friendId is numeric using the helper method
      final numericFriendId = _ensureNumericFriendId(friendId);
      debugPrint('🔍 Using numeric friend ID: $numericFriendId');

      // First get or create the conversation
      final conversation = await createOrGetConversation(numericFriendId);
      debugPrint('📝 Got conversation ID: ${conversation.id}');

      // Mark the conversation as read
      final response = await _apiService.post(
        ApiConfig.getFullUrl(
          '${ApiConfig.conversationsEndpoint}/${conversation.id}/mark_read',
        ),
      );

      if (response.statusCode == 200) {
        debugPrint('✅ Successfully marked messages as read');

        // Update local message state
        _messages = _messages.map((message) {
          if (message.sender == numericFriendId) {
            message.isRead = true;
          }
          return message;
        }).toList();
        _messagesController.add(_messages);

        // Get updated conversations list to reflect read status
        await getRecentConversations();

        // Update unread count and notify listeners
        final unreadCount = await getUnreadMessageCount();
        debugPrint('📊 New unread count: $unreadCount');
        _unreadCountController.add(unreadCount);

        return true;
      } else {
        debugPrint('❌ Failed to mark messages as read: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      debugPrint('❌ Error marking messages as read: $e');
      return false;
    }
  }

  Future<bool> sendDirectMessage({
    required String receiverId,
    required String senderUsername,
    required String receiverUsername,
    required String content,
  }) async {
    if (_channel == null) {
      debugPrint('❌ Cannot send message - WebSocket not connected');
      return false;
    }

    try {
      // Ensure receiverId is numeric using the helper method
      final numericId = _ensureNumericFriendId(receiverId);

      debugPrint(
        '📤 Attempting to send message via WebSocket to $numericId: "$content"',
      );

      // For WebSocket, format message for the backend consumer
      final messageData = {'message': content};
      _channel!.sink.add(jsonEncode(messageData));

      debugPrint('📤 Message sent via WebSocket: $messageData');

      // Add a delay to wait for the message to be processed
      await Future.delayed(Duration(milliseconds: 300));

      return true;
    } catch (e) {
      debugPrint('❌ Error sending message via WebSocket: $e');

      // Fallback to HTTP API if WebSocket fails
      try {
        debugPrint('⚡ Falling back to HTTP API for sending message');
        final numericId = _ensureNumericFriendId(receiverId);
        final data = {
          'receiver': numericId,
          'content': content,
          'sender_username': senderUsername,
          'receiver_username': receiverUsername,
        };

        final response = await _apiService.post(
          ApiConfig.getFullUrl('${ApiConfig.chatEndpoints}/messages'),
          data: data,
        );

        if (response.statusCode == 201 || response.statusCode == 200) {
          debugPrint('✅ Message sent successfully via HTTP API');
          debugPrint('API response: ${response.data}');
          return true;
        } else {
          debugPrint(
            '❌ Failed to send message via HTTP: ${response.statusCode}',
          );
          debugPrint('Error response: ${response.data}');
          return false;
        }
      } catch (httpError) {
        debugPrint('❌ Error sending message via HTTP: $httpError');
        return false;
      }
    }
  }

  void _startPingTimer() {
    // Cancel any existing timer
    _pingTimer?.cancel();

    // Send a ping every 30 seconds to keep the connection alive
    _pingTimer = Timer.periodic(const Duration(seconds: 30), (timer) {
      if (_channel != null) {
        try {
          _channel!.sink.add(jsonEncode({'type': 'ping'}));
          debugPrint('📤 Ping sent to keep connection alive');
        } catch (e) {
          debugPrint('❌ Error sending ping: $e');
          timer.cancel();
        }
      } else {
        // Cancel timer if channel is closed
        timer.cancel();
      }
    });
  }
}
