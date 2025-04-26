import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/features/authentication/pages/forgot_password_page.dart';
import 'package:echo_journal1/features/authentication/pages/login_page.dart';
import 'package:echo_journal1/features/widgets/navbar.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_journal1/services/auth/session_management.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';
import 'package:echo_journal1/services/auth/login_service.dart';
import 'package:echo_journal1/features/home/analytics/analytics_page.dart';
import 'package:echo_journal1/features/subscription/subscription_plans_page.dart';
import 'package:echo_journal1/features/settings/account_settings_page.dart';
import 'package:echo_journal1/core/providers/subscription_provider.dart';
import 'package:echo_journal1/features/admin/pages/dashboard_page.dart';
import 'package:echo_journal1/providers/leaderboard_provider.dart';
import 'package:echo_journal1/services/streak/streak_service.dart';

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
        ChangeNotifierProvider(
          create: (_) => LeaderboardProvider(StreakService()),
        ),
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
          case '/forgot_password':
            return MaterialPageRoute(
                builder: (_) => const ForgotPasswordPage());
          case '/subscription_plans':
            return MaterialPageRoute(
                builder: (_) => const SubscriptionPlansPage());
          case '/mood_analysis':
            return MaterialPageRoute(builder: (_) => const AnalyticsPage());
          case '/account_settings':
            return MaterialPageRoute(
                builder: (_) => const AccountSettingsPage());
          // Admin routes
          case '/admin':
            return MaterialPageRoute(
                builder: (_) => const AdminDashboardPage());
          case '/admin/users':
            // TODO: Implement Users page
            return MaterialPageRoute(
                builder: (_) => const AdminDashboardPage());
          case '/admin/journals':
            // TODO: Implement Journals page
            return MaterialPageRoute(
                builder: (_) => const AdminDashboardPage());
          case '/admin/subscriptions':
            // TODO: Implement Subscriptions page
            return MaterialPageRoute(
                builder: (_) => const AdminDashboardPage());
          case '/admin/analytics':
            // TODO: Implement Analytics page
            return MaterialPageRoute(
                builder: (_) => const AdminDashboardPage());
          default:
            return MaterialPageRoute(builder: (_) => const NavBar());
        }
      },
    );
  }
}
