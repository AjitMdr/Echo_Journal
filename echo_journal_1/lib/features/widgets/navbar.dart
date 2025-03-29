import 'package:echo_journal_1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal_1/features/home/journal/journal_page.dart';
import 'package:echo_journal_1/features/home/friends/friends_page.dart';
import 'package:echo_journal_1/features/home/chat/chat_page.dart';
import 'package:echo_journal_1/features/home/mood/mood_page.dart';
import 'package:echo_journal_1/features/settings/settings_page.dart';
import 'package:echo_journal_1/services/chat/chat_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'dart:async';
import 'package:echo_journal_1/features/home/analytics/analytics_page.dart';
import 'package:echo_journal_1/services/auth/secure_storage_service.dart';

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
    _pageController.jumpToPage(index);
  }

  void _toggleChatVisibility() {
    _pageController.jumpToPage(4); // Jump to chat page at index 4
    setState(() {
      _selectedIndex = 4;
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
    final themeProvider = Provider.of<ThemeProvider>(context);
    bool isDarkMode = themeProvider.isDarkMode;

    return WillPopScope(
      onWillPop: _onWillPop,
      child: Scaffold(
        body: Column(
          children: [
            Container(
              color: isDarkMode
                  ? Colors.black
                  : Color.fromARGB(255, 245, 242, 235),
              padding: const EdgeInsets.symmetric(vertical: 0),
              child: Column(
                children: [
                  AppBar(
                    backgroundColor: isDarkMode
                        ? Colors.black
                        : Color.fromARGB(255, 245, 242, 235),
                    elevation: 0,
                    automaticallyImplyLeading: false,
                    title: Text(
                      'echo',
                      style: TextStyle(
                        color: isDarkMode
                            ? Color.fromARGB(255, 245, 242, 235)
                            : const Color.fromARGB(255, 255, 87, 87),
                        fontWeight: FontWeight.bold,
                        fontSize: 24,
                      ),
                    ),
                    actions: [
                      CircleAvatar(
                        backgroundImage: profileImageUrl != null
                            ? NetworkImage(profileImageUrl!)
                            : const AssetImage('assets/images/image.png')
                                as ImageProvider,
                        radius: 18,
                      ),
                      const SizedBox(width: 8),
                    ],
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _buildNavItem(Icons.home, 0, isDarkMode, 'Home'),
                      _buildNavItem(Icons.people, 1, isDarkMode, 'Friends'),
                      _buildNavItem(
                        Icons.bar_chart,
                        2,
                        isDarkMode,
                        'Analytics',
                      ),
                      _buildNavItem(
                        FontAwesomeIcons.faceSmile,
                        3,
                        isDarkMode,
                        'Mood',
                      ),
                      _buildNavItem(Icons.chat, 4, isDarkMode, 'Chat'),
                      _buildNavItem(Icons.menu, 5, isDarkMode, 'Menu'),
                    ],
                  ),
                ],
              ),
            ),
            Expanded(
              child: PageView(
                controller: _pageController,
                onPageChanged: _onPageChanged,
                physics:
                    const CustomPageViewScrollPhysics(), // Custom physics to control swipe
                children: [
                  JournalPage(),
                  FriendsPage(),
                  const AnalyticsPage(),
                  MoodPage(),
                  ChatPage(), // Chat page at index 4
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
    String tooltip,
  ) {
    return Tooltip(
      message: tooltip,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Stack(
            children: [
              IconButton(
                icon: Icon(
                  icon,
                  color: _selectedIndex == index && !_isChatVisible
                      ? const Color.fromARGB(255, 255, 87, 87)
                      : isDarkMode
                          ? Color.fromARGB(255, 245, 242, 235)
                          : Colors.black54,
                ),
                onPressed: () {
                  _onItemTapped(index);
                  if (_isChatVisible) {
                    setState(() => _isChatVisible = false);
                  }
                },
              ),
              if (index == 4 &&
                  _unreadMessageCount > 0) // Index 4 is for chat now
                Positioned(
                  right: 5,
                  top: 5,
                  child: Container(
                    padding: EdgeInsets.all(2),
                    decoration: BoxDecoration(
                      color: Colors.red,
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: isDarkMode
                            ? Colors.black
                            : Color.fromARGB(255, 245, 242, 235),
                        width: 1.5,
                      ),
                    ),
                    constraints: BoxConstraints(minWidth: 14, minHeight: 14),
                    child: Text(
                      _unreadMessageCount > 99
                          ? '99+'
                          : _unreadMessageCount.toString(),
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
            ],
          ),
          if (_selectedIndex == index && !_isChatVisible)
            Container(
              height: 3,
              width: 20,
              color: const Color.fromARGB(255, 255, 87, 87),
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
