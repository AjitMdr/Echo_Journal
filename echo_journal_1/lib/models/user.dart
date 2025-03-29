class User {
  final int id;
  final String username;
  final String email;
  final String? profileImage;
  final bool? isOnline;
  final DateTime? lastSeen;
  final String? bio;
  final DateTime createdAt;
  final DateTime updatedAt;

  User({
    required this.id,
    required this.username,
    required this.email,
    this.profileImage,
    this.isOnline,
    this.lastSeen,
    this.bio,
    required this.createdAt,
    required this.updatedAt,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    try {
      return User(
        id: json['id'] ?? 0,
        username: json['username'] ?? 'Unknown',
        email: json['email'] ?? '',
        profileImage: json['profile_picture'],
        isOnline: json['is_online'],
        lastSeen: json['last_seen'] != null
            ? DateTime.parse(json['last_seen'])
            : null,
        bio: json['bio'],
        createdAt: DateTime.parse(
            json['created_at'] ?? DateTime.now().toIso8601String()),
        updatedAt: DateTime.parse(
            json['updated_at'] ?? DateTime.now().toIso8601String()),
      );
    } catch (e) {
      print('Error parsing User data: $e');
      print('JSON data: $json');
      // Return a default user object in case of parsing errors
      return User(
        id: 0,
        username: 'Unknown',
        email: '',
        createdAt: DateTime.now(),
        updatedAt: DateTime.now(),
      );
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'username': username,
      'email': email,
      'profile_picture': profileImage,
      'is_online': isOnline,
      'last_seen': lastSeen?.toIso8601String(),
      'bio': bio,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }
}
