import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter/foundation.dart';
import 'dart:math' as math;

/// SecureStorageService: A service class to handle secure storage of sensitive data
/// This includes tokens and user authentication information using Flutter Secure Storage
class SecureStorageService {
  // Storage instance with secure options
  static final _storage = FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
  );

  // Key constants for stored values
  static const String _keyAccessToken = 'access_token';
  static const String _keyRefreshToken = 'refresh_token';
  static const String _keyUserId = 'user_id';
  static const String _keyUsername = 'username';
  static const String _keyUserEmail = 'user_email';
  static const String _keyLoginTimestamp = 'login_timestamp';

  // Cache for frequently accessed values
  static String? _cachedAccessToken;
  static String? _cachedRefreshToken;
  static String? _cachedUserId;
  static String? _cachedUsername;
  static String? _cachedUserEmail;
  static int? _cachedLoginTimestamp;

  /// Initialize the service and load cached values
  static Future<void> initialize() async {
    try {
      _cachedAccessToken = await _storage.read(key: _keyAccessToken);
      _cachedRefreshToken = await _storage.read(key: _keyRefreshToken);
      _cachedUserId = await _storage.read(key: _keyUserId);
      _cachedUsername = await _storage.read(key: _keyUsername);
      _cachedUserEmail = await _storage.read(key: _keyUserEmail);
      final timestampStr = await _storage.read(key: _keyLoginTimestamp);
      _cachedLoginTimestamp =
          timestampStr != null ? int.tryParse(timestampStr) : null;
    } catch (e) {
      debugPrint('❌ Error initializing secure storage: $e');
    }
  }

  /// Saves authentication tokens and user data securely
  static Future<void> saveAuthData({
    required String accessToken,
    required String refreshToken,
    required String userId,
    required String username,
    required String email,
  }) async {
    // Update cache
    _cachedAccessToken = accessToken;
    _cachedRefreshToken = refreshToken;
    _cachedUserId = userId;
    _cachedUsername = username;
    _cachedUserEmail = email;
    _cachedLoginTimestamp = DateTime.now().millisecondsSinceEpoch;

    // Save to storage
    await Future.wait([
      _storage.write(key: _keyAccessToken, value: accessToken),
      _storage.write(key: _keyRefreshToken, value: refreshToken),
      _storage.write(key: _keyUserId, value: userId),
      _storage.write(key: _keyUsername, value: username),
      _storage.write(key: _keyUserEmail, value: email),
      _storage.write(
          key: _keyLoginTimestamp, value: _cachedLoginTimestamp.toString()),
    ]);

    debugPrint('🔒 Auth data securely saved: tokens and user info');
  }

  /// Retrieves the stored access token
  static Future<String?> getAccessToken() async {
    if (_cachedAccessToken != null) return _cachedAccessToken;
    _cachedAccessToken = await _storage.read(key: _keyAccessToken);
    return _cachedAccessToken;
  }

  /// Retrieves the stored refresh token
  static Future<String?> getRefreshToken() async {
    if (_cachedRefreshToken != null) return _cachedRefreshToken;
    _cachedRefreshToken = await _storage.read(key: _keyRefreshToken);
    return _cachedRefreshToken;
  }

  /// Gets the current user ID
  static Future<String?> getUserId() async {
    if (_cachedUserId != null) return _cachedUserId;
    _cachedUserId = await _storage.read(key: _keyUserId);
    return _cachedUserId;
  }

  /// Gets the current username
  static Future<String?> getUsername() async {
    if (_cachedUsername != null) return _cachedUsername;
    _cachedUsername = await _storage.read(key: _keyUsername);
    return _cachedUsername;
  }

  /// Gets the user's email
  static Future<String?> getUserEmail() async {
    if (_cachedUserEmail != null) return _cachedUserEmail;
    _cachedUserEmail = await _storage.read(key: _keyUserEmail);
    return _cachedUserEmail;
  }

  /// Gets login timestamp for session management
  static Future<int?> getLoginTimestamp() async {
    if (_cachedLoginTimestamp != null) return _cachedLoginTimestamp;
    final timestampStr = await _storage.read(key: _keyLoginTimestamp);
    _cachedLoginTimestamp =
        timestampStr != null ? int.tryParse(timestampStr) : null;
    return _cachedLoginTimestamp;
  }

  /// Get all user data as a map
  static Future<Map<String, String?>> getUserData() async {
    return {
      'user_id': await getUserId(),
      'username': await getUsername(),
      'email': await getUserEmail(),
    };
  }

  /// Clears all stored authentication data (logout)
  static Future<void> clearAuthData() async {
    // Clear cache
    _cachedAccessToken = null;
    _cachedRefreshToken = null;
    _cachedUserId = null;
    _cachedUsername = null;
    _cachedUserEmail = null;
    _cachedLoginTimestamp = null;

    // Clear storage
    await _storage.deleteAll();
    debugPrint('🔒 All secure auth data cleared');
  }

  /// Checks if the user is logged in by confirming access token exists
  static Future<bool> isLoggedIn() async {
    if (_cachedAccessToken != null) return true;
    _cachedAccessToken = await _storage.read(key: _keyAccessToken);
    return _cachedAccessToken != null;
  }

  /// Debug method to print all stored values - useful for troubleshooting
  static Future<void> debugPrintAllValues() async {
    print('🔍 DEBUG - SECURE STORAGE CONTENTS:');
    final token = await getAccessToken();
    print(
      '🔑 Access Token: ${token != null ? "EXISTS (${token.substring(0, math.min(10, token.length))}...)" : "NULL"}',
    );
    print(
      '🔄 Refresh Token: ${await getRefreshToken() != null ? "EXISTS" : "NULL"}',
    );
    print('👤 User ID: ${await getUserId()}');
    print('👤 Username: ${await getUsername()}');
    print('📧 Email: ${await getUserEmail()}');
    print('⏰ Login Timestamp: ${await getLoginTimestamp()}');
  }
}
