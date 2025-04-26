import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/features/home/journal/journal_page.dart';
import 'package:echo_journal1/features/home/friends/friends_page.dart';
import 'package:echo_journal1/features/home/chat/chat_page.dart';
import 'package:echo_journal1/features/home/mood/mood_page.dart';
import 'package:echo_journal1/features/settings/settings_page.dart';
import 'package:echo_journal1/services/chat/chat_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'dart:async';
import 'package:echo_journal1/features/home/analytics/analytics_page.dart';
import 'package:echo_journal1/services/auth/secure_storage_service.dart';
import 'package:echo_journal1/core/providers/subscription_provider.dart';
import 'package:echo_journal1/features/subscription/subscription_plans_page.dart';
import 'package:echo_journal1/features/home/leaderboard/leaderboard_page.dart';

class NavBar extends StatefulWidget {
  const NavBar({super.key});

  @override
  NavBarState createState() => NavBarState();
}

class NavBarState extends State<NavBar> {
  int _selectedIndex = 0;
  final PageController _pageController = PageController();
  String? profileImageUrl;
  DateTime? _lastBackPressTime;
  bool _isChatVisible = false;
  int _unreadMessageCount = 0;
  Timer? _unreadMessageTimer;
  final ChatService _chatService = ChatService();
  StreamSubscription? _unreadCountSubscription;

  @override
  void initState() {
    super.initState();
    // Prevent system back button from popping to login
    SystemChannels.platform.invokeMethod(
      'SystemNavigator.setNavigationEnabledForAndroid',
      false,
    );

    // Load unread message count initially
    _loadUnreadMessageCount();

    // Subscribe to unread count stream
    _unreadCountSubscription = _chatService.unreadCount.listen(
      (count) {
        if (mounted) {
          setState(() {
            _unreadMessageCount = count;
            debugPrint('NavBar: Updated unread count from stream: $count');
          });
        }
      },
      onError: (error) {
        debugPrint('NavBar: Error from unread count stream: $error');
      },
    );

    // Set up timer to refresh unread count as backup
    _unreadMessageTimer = Timer.periodic(
      const Duration(seconds: 30),
      (_) => _loadUnreadMessageCount(),
    );

    // Initialize subscription provider
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final subscriptionProvider =
          Provider.of<SubscriptionProvider>(context, listen: false);
      subscriptionProvider.checkSubscription();
    });
  }

  @override
  void dispose() {
    _unreadMessageTimer?.cancel();
    _unreadCountSubscription?.cancel();
    super.dispose();
  }

  Future<void> _loadUnreadMessageCount() async {
    try {
      final count = await _chatService.getUnreadMessageCount();
      debugPrint('NavBar: Unread message count: $count');
      if (mounted) {
        setState(() {
          _unreadMessageCount = count;
          debugPrint(
              'NavBar: Updated unread count state: $_unreadMessageCount');
        });
      }
    } catch (e) {
      debugPrint('Error loading unread message count: $e');
    }
  }

  void _onPageChanged(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  void _onItemTapped(int index) {
    final subscriptionProvider =
        Provider.of<SubscriptionProvider>(context, listen: false);

    // Check if user is trying to access mood analysis
    if (index == 4) {
      if (!subscriptionProvider.isPremium) {
        Navigator.push(
          context,
          MaterialPageRoute(
              builder: (context) => const SubscriptionPlansPage()),
        );
        return;
      }
    }

    _pageController.jumpToPage(index);
  }

  void _toggleChatVisibility() {
    _pageController.jumpToPage(5); // Jump to chat page at index 5
    setState(() {
      _selectedIndex = 5;
      _isChatVisible = false; // No longer need this toggle mechanism
      // Refresh unread count when opening chat
      _loadUnreadMessageCount();
    });
  }

  Future<bool> _onWillPop() async {
    if (_isChatVisible) {
      // If chat is visible, hide it first
      setState(() {
        _isChatVisible = false;
        _pageController.jumpToPage(_selectedIndex);
      });
      return false;
    }

    if (_selectedIndex != 0) {
      // If not on home page, go to home page
      _pageController.jumpToPage(0);
      setState(() {
        _selectedIndex = 0;
      });
      return false;
    }

    // If on home page, implement double-back-tap to exit
    final now = DateTime.now();
    if (_lastBackPressTime == null ||
        now.difference(_lastBackPressTime!) > const Duration(seconds: 2)) {
      _lastBackPressTime = now;
      ToastHelper.showInfo(context, 'Press back again to exit');
      return false;
    }
    return true; // Allow app exit on second tap
  }

  Future<void> _logout() async {
    // Use SecureStorageService to clear all auth data
    await SecureStorageService.clearAuthData();

    // Also clear SharedPreferences for backward compatibility
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('access_token');
    await prefs.remove('refresh_token');
    await prefs.remove('user_id');

    if (mounted) {
      ToastHelper.showInfo(context, 'Logged out successfully');
      Navigator.of(context).pushReplacementNamed('/login');
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;
    final subscriptionProvider = Provider.of<SubscriptionProvider>(context);

    return WillPopScope(
      onWillPop: _onWillPop,
      child: Scaffold(
        body: Column(
          children: [
            Container(
              decoration: BoxDecoration(
                color: isDarkMode ? Colors.grey[900] : Colors.white,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: SafeArea(
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildNavItem(Icons.home, 0, isDarkMode, 'Home'),
                        _buildNavItem(Icons.people, 1, isDarkMode, 'Friends'),
                        _buildNavItem(
                            Icons.emoji_events, 2, isDarkMode, 'Leaderboard'),
                        _buildNavItem(
                          Icons.bar_chart,
                          3,
                          isDarkMode,
                          'Analytics',
                          isPremiumFeature: !subscriptionProvider.isPremium,
                        ),
                        _buildNavItem(
                          Icons.mood,
                          4,
                          isDarkMode,
                          'Mood',
                        ),
                        _buildNavItem(
                          Icons.chat,
                          5,
                          isDarkMode,
                          'Chat',
                          badge: _unreadMessageCount > 0
                              ? _unreadMessageCount.toString()
                              : null,
                        ),
                        _buildNavItem(
                          Icons.menu,
                          6,
                          isDarkMode,
                          'Menu',
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            Expanded(
              child: PageView(
                controller: _pageController,
                onPageChanged: _onPageChanged,
                physics: const CustomPageViewScrollPhysics(),
                children: [
                  JournalPage(),
                  FriendsPage(),
                  LeaderboardPage(),
                  subscriptionProvider.isPremium
                      ? const AnalyticsPage()
                      : const SubscriptionPlansPage(),
                  MoodPage(),
                  ChatPage(),
                  const SettingsPage(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNavItem(
    IconData icon,
    int index,
    bool isDarkMode,
    String tooltip, {
    String? badge,
    bool isPremiumFeature = false,
  }) {
    final isSelected = index == _selectedIndex;
    return Tooltip(
      message: tooltip + (isPremiumFeature ? ' (Premium)' : ''),
      child: Stack(
        children: [
          IconButton(
            icon: Stack(
              children: [
                Icon(
                  icon,
                  color: isSelected
                      ? Theme.of(context).primaryColor
                      : isDarkMode
                          ? Colors.white54
                          : Colors.black54,
                ),
                if (isPremiumFeature)
                  Positioned(
                    right: -2,
                    top: -2,
                    child: Icon(
                      Icons.star,
                      size: 12,
                      color: Colors.amber,
                    ),
                  ),
              ],
            ),
            onPressed: () => _onItemTapped(index),
          ),
          if (badge != null)
            Positioned(
              right: 0,
              top: 0,
              child: Container(
                padding: const EdgeInsets.all(2),
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(10),
                ),
                constraints: const BoxConstraints(
                  minWidth: 20,
                  minHeight: 20,
                ),
                child: Text(
                  badge,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// Custom ScrollPhysics to control page swiping
class CustomPageViewScrollPhysics extends ScrollPhysics {
  const CustomPageViewScrollPhysics({super.parent});

  @override
  CustomPageViewScrollPhysics applyTo(ScrollPhysics? ancestor) {
    return CustomPageViewScrollPhysics(parent: buildParent(ancestor));
  }

  @override
  double get dragStartDistanceMotionThreshold => 20.0;
}
