import 'package:flutter/foundation.dart';
import 'message.dart';
import 'user.dart';

class Conversation {
  final String id;
  final List<User> participants;
  final DateTime createdAt;
  final DateTime updatedAt;
  final Message? lastMessage;
  final int unreadCount;

  Conversation({
    required this.id,
    required this.participants,
    required this.createdAt,
    required this.updatedAt,
    this.lastMessage,
    required this.unreadCount,
  });

  factory Conversation.fromJson(Map<String, dynamic> json) {
    try {
      debugPrint('🔍 Parsing conversation with ID: ${json['id']}');

      // Ensure participants is a List
      List<dynamic> participantsList = [];

      try {
        if (json['participants'] is List) {
          participantsList = json['participants'] as List;
          debugPrint('👥 Found ${participantsList.length} participants');
        } else {
          debugPrint(
            '⚠️ Participants is not a list: ${json['participants']?.runtimeType}',
          );
        }
      } catch (e) {
        debugPrint('❌ Error extracting participants: $e');
      }

      return Conversation(
        id: json['id']?.toString() ?? '',
        participants:
            participantsList.map((p) {
              try {
                if (p is Map) {
                  // Convert to Map<String, dynamic>
                  final Map<String, dynamic> userMap = {};
                  p.forEach((key, value) {
                    userMap[key.toString()] = value;
                  });
                  return User.fromJson(userMap);
                } else {
                  debugPrint('⚠️ Participant is not a Map: ${p?.runtimeType}');
                  return User(id: '', username: 'Unknown', email: '');
                }
              } catch (e) {
                debugPrint('❌ Error parsing participant: $e');
                return User(id: '', username: 'Unknown', email: '');
              }
            }).toList(),
        createdAt:
            json['created_at'] != null
                ? DateTime.parse(json['created_at'].toString())
                : DateTime.now(),
        updatedAt:
            json['updated_at'] != null
                ? DateTime.parse(json['updated_at'].toString())
                : DateTime.now(),
        lastMessage:
            json['last_message'] != null
                ? Message.fromJson(json['last_message'])
                : null,
        unreadCount:
            json['unread_count'] != null
                ? int.tryParse(json['unread_count'].toString()) ?? 0
                : 0,
      );
    } catch (e) {
      debugPrint('❌ Error parsing Conversation: $e');
      debugPrint('JSON data: $json');
      // Return a fallback conversation
      return Conversation(
        id: '',
        participants: [],
        createdAt: DateTime.now(),
        updatedAt: DateTime.now(),
        lastMessage: null,
        unreadCount: 0,
      );
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'participants': participants.map((p) => p.toJson()).toList(),
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
      'last_message': lastMessage?.toJson(),
      'unread_count': unreadCount,
    };
  }

  User getOtherParticipant(String userId) {
    try {
      return participants.firstWhere(
        (p) => p.id != userId,
        orElse: () => User(id: '', username: 'Unknown', email: ''),
      );
    } catch (e) {
      debugPrint('Error getting other participant: $e');
      return User(id: '', username: 'Unknown', email: '');
    }
  }
}
