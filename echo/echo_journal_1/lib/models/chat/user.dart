import 'package:flutter/foundation.dart';

class User {
  final String id;
  final String username;
  final String email;
  final String? profilePictureUrl;

  User({
    required this.id,
    required this.username,
    required this.email,
    this.profilePictureUrl,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    try {
      return User(
        id: json['id']?.toString() ?? '',
        username: json['username']?.toString() ?? 'Unknown',
        email: json['email']?.toString() ?? '',
        profilePictureUrl: json['profile_picture_url']?.toString(),
      );
    } catch (e) {
      debugPrint('Error parsing User: $e');
      debugPrint('JSON data: $json');
      return User(id: '', username: 'Unknown', email: '');
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'username': username,
      'email': email,
      'profile_picture_url': profilePictureUrl,
    };
  }
}
