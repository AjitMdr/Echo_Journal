import 'package:flutter/foundation.dart';

class Message {
  final String id;
  final String sender;
  final String senderUsername;
  final String receiver;
  final String receiverUsername;
  final String content;
  final String message;
  final DateTime timestamp;
  bool isRead;
  final String? conversationId;
  final String type;

  Message({
    required this.id,
    required this.sender,
    required this.senderUsername,
    required this.receiver,
    required this.receiverUsername,
    required this.content,
    required this.message,
    required this.timestamp,
    required this.isRead,
    this.conversationId,
    required this.type,
  });

  factory Message.fromJson(Map<String, dynamic> json) {
    try {
      // Handle potential array input
      final data = json is List ? (json.isNotEmpty ? json[0] : {}) : json;

      // If data is still not a map, create an empty one
      final Map<String, dynamic> messageData =
          data is Map ? Map<String, dynamic>.from(data) : {};

      debugPrint('Processing Message from JSON: ${messageData.keys}');

      // Parse timestamp
      DateTime messageTime;
      try {
        if (messageData.containsKey('timestamp') &&
            messageData['timestamp'] != null) {
          messageTime = DateTime.parse(messageData['timestamp'].toString());
        } else if (messageData.containsKey('created_at') &&
            messageData['created_at'] != null) {
          messageTime = DateTime.parse(messageData['created_at'].toString());
        } else {
          messageTime = DateTime.now();
        }
      } catch (e) {
        debugPrint('Error parsing timestamp: $e');
        messageTime = DateTime.now();
      }

      return Message(
        id: messageData['id']?.toString() ?? '',
        sender: messageData['sender']?.toString() ?? '',
        senderUsername: messageData['sender_username']?.toString() ?? '',
        receiver: messageData['receiver']?.toString() ?? '',
        receiverUsername: messageData['receiver_username']?.toString() ?? '',
        content: messageData['content']?.toString() ?? '',
        message:
            messageData['message']?.toString() ??
            messageData['content']?.toString() ??
            '',
        timestamp: messageTime,
        isRead:
            messageData['is_read'] == true ||
            messageData['is_read'] == 1 ||
            messageData['is_read'] == '1' ||
            messageData['is_read'] == 'true',
        conversationId: messageData['conversation_id']?.toString(),
        type: messageData['type']?.toString() ?? 'chat_message',
      );
    } catch (e) {
      debugPrint('Error parsing Message: $e');
      debugPrint('JSON data: $json');
      return Message(
        id: '',
        sender: '',
        senderUsername: '',
        receiver: '',
        receiverUsername: '',
        content: 'Error loading message',
        message: 'Error loading message',
        timestamp: DateTime.now(),
        isRead: false,
        type: 'chat_message',
      );
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'sender': sender,
      'sender_username': senderUsername,
      'receiver': receiver,
      'receiver_username': receiverUsername,
      'content': content,
      'message': message,
      'timestamp': timestamp.toIso8601String(),
      'is_read': isRead,
      'conversation_id': conversationId,
      'type': type,
    };
  }
}
