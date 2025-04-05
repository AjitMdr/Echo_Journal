import 'package:echo_fe/core/configs/theme/theme-provider.dart';
import 'package:echo_fe/features/authentication/pages/login_page.dart';
import 'package:echo_fe/features/widgets/navbar.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_fe/services/auth/session_management.dart';
import 'package:echo_fe/services/auth/secure_storage_service.dart';
import 'package:echo_fe/services/auth/login_service.dart';
import 'package:echo_fe/features/home/analytics/analytics_page.dart';
import 'package:echo_fe/features/subscription/subscription_plans_page.dart';
import 'core/providers/subscription_provider.dart';

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
    MultiProvider(
      providers: [
        ChangeNotifierProvider(
            create: (_) => ThemeProvider(isDarkMode: isDarkMode)),
        ChangeNotifierProvider(create: (_) => SubscriptionProvider()),
      ],
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
    try {
      final isAuthenticated = await SessionManager.initializeSilently();

      if (mounted) {
        setState(() {
          _isLoggedIn = isAuthenticated;
          _isInitialized = true;
        });

        if (isAuthenticated) {
          final subscriptionProvider =
              Provider.of<SubscriptionProvider>(context, listen: false);
          await subscriptionProvider.checkSubscription();
        }
      }
    } catch (e) {
      debugPrint('❌ Error during auth initialization: $e');
      if (mounted) {
        setState(() {
          _isLoggedIn = false;
          _isInitialized = true;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);

    Widget home;
    if (!_isInitialized) {
      home = const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    } else {
      home = _isLoggedIn ? const NavBar() : const LoginPage();
    }

    return MaterialApp(
      title: 'Echo Journal',
      theme: themeProvider.themeData,
      debugShowCheckedModeBanner: false,
      home: home,
      onGenerateRoute: (settings) {
        if (!_isInitialized || !_isLoggedIn) {
          return MaterialPageRoute(builder: (_) => const LoginPage());
        }

        switch (settings.name) {
          case '/':
            return MaterialPageRoute(builder: (_) => const NavBar());
          case '/login':
            return MaterialPageRoute(builder: (_) => const LoginPage());
          case '/subscription/plans':
            return MaterialPageRoute(
                builder: (_) => const SubscriptionPlansPage());
          case '/mood_analysis':
            return MaterialPageRoute(builder: (_) => const AnalyticsPage());
          default:
            return MaterialPageRoute(builder: (_) => const NavBar());
        }
      },
    );
  }
}
