// lib/services/auth/session_manager.dart

import 'dart:async';
import 'package:shared_preferences/shared_preferences.dart';
import 'login_service.dart';

/// 🔐 Service class handling session management including session timeouts,
/// validation, and persistence across app restarts.
class SessionManager {
  /// ⏱️ Duration in minutes before a session expires
  static const int sessionDurationMinutes = 60;

  /// 🔑 Key for storing login timestamp in SharedPreferences
  static const String _keyLoginTimestamp = 'login_timestamp';

  /// ⏲️ Timer for managing session expiration
  static Timer? _sessionTimer;

  /// 📡 Callback function for session state changes
  static Function(bool)? _onSessionStateChanged;

  /// 🚀 Initialize session manager and validate existing session
  ///
  /// Takes a callback function that will be called whenever session state changes
  static Future<void> initialize(Function(bool) onSessionStateChanged) async {
    _onSessionStateChanged = onSessionStateChanged;
    final isValid = await checkSession();
    _onSessionStateChanged?.call(isValid);
    print('🔄 Session Manager initialized');
  }

  /// ✨ Start a new session
  ///
  /// Creates a new timestamp and starts the session timer
  static Future<void> startSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(
        _keyLoginTimestamp, DateTime.now().millisecondsSinceEpoch);
    _startSessionTimer();
    _onSessionStateChanged?.call(true);
    print('✅ New session started');
    await AuthService.printSharedPreferences();
  }

  /// 🔍 Check if current session is valid
  ///
  /// Returns true if session exists and hasn't expired
  static Future<bool> checkSession() async {
    final prefs = await SharedPreferences.getInstance();
    final loginTimestamp = prefs.getInt(_keyLoginTimestamp);
    final isLoggedIn = await AuthService.isLoggedIn();

    if (loginTimestamp == null || !isLoggedIn) {
      print('❌ No valid session found');
      return false;
    }

    final currentTime = DateTime.now().millisecondsSinceEpoch;
    final sessionAge =
        (currentTime - loginTimestamp) / (1000 * 60); // Convert to minutes

    if (sessionAge < sessionDurationMinutes) {
      _startSessionTimer();
      print(
          '✅ Valid session found, age: ${sessionAge.toStringAsFixed(2)} minutes');
      return true;
    } else {
      print(
          '⚠️ Session expired after ${sessionAge.toStringAsFixed(2)} minutes');
      await endSession();
      return false;
    }
  }

  /// 🚪 End current session and logout user
  ///
  /// Clears session data and cancels the session timer
  static Future<void> endSession() async {
    await AuthService.logout();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keyLoginTimestamp);
    _sessionTimer?.cancel();
    _onSessionStateChanged?.call(false);
    print('🚪 Session ended and user logged out');
  }

  /// ⏲️ Start or restart the session timer
  ///
  /// Called internally to manage session timeout
  static void _startSessionTimer() {
    _sessionTimer?.cancel();
    _sessionTimer = Timer(Duration(minutes: sessionDurationMinutes), () {
      endSession();
      print('⏰ Session timeout - auto logout triggered');
    });
    print('⏱️ Session timer started/reset');
  }

  /// 📊 Get current session information
  ///
  /// Returns a Map containing session status and remaining time
  static Future<Map<String, dynamic>> getSessionInfo() async {
    final prefs = await SharedPreferences.getInstance();
    final loginTimestamp = prefs.getInt(_keyLoginTimestamp);
    final isLoggedIn = await AuthService.isLoggedIn();

    if (loginTimestamp == null || !isLoggedIn) {
      return {
        'isValid': false,
        'remainingMinutes': 0,
        'sessionAge': 0,
      };
    }

    final currentTime = DateTime.now().millisecondsSinceEpoch;
    final sessionAge = (currentTime - loginTimestamp) / (1000 * 60);
    final remainingMinutes = sessionDurationMinutes - sessionAge;

    return {
      'isValid': remainingMinutes > 0,
      'remainingMinutes': remainingMinutes.round(),
      'sessionAge': sessionAge.round(),
    };
  }
}
