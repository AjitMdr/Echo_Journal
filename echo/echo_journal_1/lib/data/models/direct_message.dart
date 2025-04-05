import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';

class DirectMessage {
  final String id;
  final String sender;
  final String receiver;
  final String content;
  final DateTime timestamp;
  bool isRead;
  final String? senderUsername;
  final String? receiverUsername;

  DirectMessage({
    required this.id,
    required this.sender,
    required this.receiver,
    required this.content,
    required this.timestamp,
    required this.isRead,
    this.senderUsername,
    this.receiverUsername,
  });

  factory DirectMessage.fromJson(Map<String, dynamic> json) {
    try {
      // Handle potential array input (common in WebSocket responses)
      final data = json is List ? (json.isNotEmpty ? json[0] : {}) : json;

      // If data is still not a map, create an empty one
      final Map<String, dynamic> messageData =
          data is Map ? Map<String, dynamic>.from(data) : {};

      debugPrint('Processing DirectMessage from JSON: ${messageData.keys}');

      // Extract content from either 'content' or 'message' field
      String messageContent = '';
      if (messageData.containsKey('content')) {
        messageContent = messageData['content']?.toString() ?? '';
      } else if (messageData.containsKey('message')) {
        messageContent = messageData['message']?.toString() ?? '';
      }

      // Extract and parse timestamp
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

      return DirectMessage(
        id: messageData['id']?.toString() ?? '0',
        sender: messageData['sender']?.toString() ?? '',
        receiver: messageData['receiver']?.toString() ?? '',
        content: messageContent,
        timestamp: messageTime,
        isRead: messageData['is_read'] == true ||
            messageData['is_read'] == 1 ||
            messageData['is_read'] == '1' ||
            messageData['is_read'] == 'true',
        senderUsername: messageData['sender_username']?.toString(),
        receiverUsername: messageData['receiver_username']?.toString(),
      );
    } catch (e) {
      debugPrint('Error parsing DirectMessage: $e');
      debugPrint('JSON data: $json');
      // Return a fallback message with default values
      return DirectMessage(
        id: '0',
        sender: '',
        receiver: '',
        content: 'Error loading message',
        timestamp: DateTime.now(),
        isRead: false,
      );
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'sender': sender,
      'receiver': receiver,
      'content': content,
      'timestamp': timestamp.toIso8601String(),
      'is_read': isRead,
      'sender_username': senderUsername,
      'receiver_username': receiverUsername,
    };
  }
}
