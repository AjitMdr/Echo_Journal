import 'package:echo_journal1/data/models/direct_message.dart';
import 'package:flutter/material.dart';

class Participant {
  final int id;
  final String username;
  final String email;

  Participant({required this.id, required this.username, required this.email});

  factory Participant.fromJson(Map<String, dynamic> json) {
    try {
      final dynamic rawId = json['id'];
      int parsedId;

      if (rawId is int) {
        parsedId = rawId;
      } else if (rawId is String) {
        parsedId = int.tryParse(rawId) ?? 0;
      } else {
        parsedId = 0;
      }

      return Participant(
        id: parsedId,
        username: json['username'] ?? '',
        email: json['email'] ?? '',
      );
    } catch (e) {
      debugPrint('Error parsing Participant: $e');
      return Participant(
        id: 0,
        username: json['username'] ?? 'Unknown',
        email: '',
      );
    }
  }

  Map<String, dynamic> toJson() {
    return {'id': id, 'username': username, 'email': email};
  }
}

class Conversation {
  final String id;
  final List<Participant> participants;
  final DateTime createdAt;
  final DateTime updatedAt;
  final DirectMessage? lastMessage;
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
      final List<Participant> participantsList = [];
      if (json['participants'] is List) {
        participantsList.addAll(
          (json['participants'] as List)
              .map((p) => Participant.fromJson(p))
              // ignore: unnecessary_null_comparison
              .where((p) => p != null)
              .toList(),
        );
      }

      return Conversation(
        id: json['id']?.toString() ?? '',
        participants: participantsList,
        createdAt: DateTime.tryParse(json['created_at']?.toString() ?? '') ??
            DateTime.now(),
        updatedAt: DateTime.tryParse(json['updated_at']?.toString() ?? '') ??
            DateTime.now(),
        lastMessage: json['last_message'] != null
            ? DirectMessage.fromJson(json['last_message'])
            : null,
        unreadCount: json['unread_count'] is int
            ? json['unread_count']
            : (int.tryParse(json['unread_count']?.toString() ?? '0') ?? 0),
      );
    } catch (e) {
      debugPrint('Error parsing Conversation: $e');
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

  Participant getOtherParticipant(String userId) {
    return participants.firstWhere(
      (p) => p.id.toString() != userId,
      orElse: () => Participant(id: 0, username: 'Unknown', email: ''),
    );
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
}
