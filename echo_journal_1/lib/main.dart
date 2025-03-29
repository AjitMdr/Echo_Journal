import 'package:echo_journal_1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal_1/features/authentication/pages/login_page.dart';
import 'package:echo_journal_1/features/widgets/navbar.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_journal_1/services/auth/session_management.dart';
import 'package:echo_journal_1/services/auth/secure_storage_service.dart';
import 'package:echo_journal_1/services/auth/login_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    await SecureStorageService.initialize();
  } catch (e) {
    debugPrint('❌ Error initializing secure storage: $e');
  }

  AuthService.initialize();

  final prefs = await SharedPreferences.getInstance();
  bool isDarkMode = prefs.getBool('isDarkMode') ?? false;

  runApp(
    ChangeNotifierProvider(
      create: (context) => ThemeProvider(isDarkMode: isDarkMode),
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool _isLoggedIn = false;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeAuth();
  }

  Future<void> _initializeAuth() async {
    // Use compute to run token verification in a separate isolate
    final isAuthenticated = await SessionManager.initializeSilently();
    if (mounted) {
      setState(() {
        _isLoggedIn = isAuthenticated;
        _isInitialized = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);

    if (!_isInitialized) {
      return MaterialApp(
        theme: themeProvider.themeData,
        debugShowCheckedModeBanner: false,
        home: const Scaffold(
          body: Center(
            child: CircularProgressIndicator(),
          ),
        ),
      );
    }

    return MaterialApp(
      title: 'Your App',
      theme: themeProvider.themeData,
      debugShowCheckedModeBanner: false,
      home: _isLoggedIn ? const NavBar() : const LoginPage(),
    );
  }
}
