import 'package:echo/page/features/home/journal/journal.dart';
import 'package:echo/page/features/home/mood_analysis/mood_analysis.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:provider/provider.dart';
import '../../settings/setting_page.dart';
import '../../../../themes/theme_provider.dart';

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

  @override
  void initState() {
    super.initState();
    // Prevent system back button from popping to login
    SystemChannels.platform
        .invokeMethod('SystemNavigator.setNavigationEnabledForAndroid', false);
  }

  void _onPageChanged(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  void _onItemTapped(int index) {
    _pageController.jumpToPage(index);
  }

  Future<bool> _onWillPop() async {
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
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Press back again to exit'),
          duration: Duration(seconds: 2),
        ),
      );
      return false;
    }
    return true; // Allow app exit on second tap
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
                      IconButton(
                        icon: Icon(Icons.chat,
                            color: isDarkMode
                                ? Color.fromARGB(255, 245, 242, 235)
                                : Colors.black),
                        onPressed: () {},
                      ),
                      IconButton(
                        icon: Icon(Icons.search,
                            color: isDarkMode
                                ? Color.fromARGB(255, 245, 242, 235)
                                : Colors.black),
                        onPressed: () {},
                      ),
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
                      _buildNavItem(Icons.home, 0, isDarkMode),
                      _buildNavItem(Icons.people, 1, isDarkMode),
                      _buildNavItem(Icons.bar_chart, 2, isDarkMode),
                      _buildNavItem(FontAwesomeIcons.faceSmile, 3, isDarkMode),
                      _buildNavItem(Icons.notifications, 4, isDarkMode),
                      _buildNavItem(Icons.menu, 5, isDarkMode),
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
                  MoodAnalysisPage(),
                  const Center(child: Text('Mood Analysis')),
                  const Center(child: Text('Analytics')),
                  const Center(child: Text('Notifications Page')),
                  const SettingsPage(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNavItem(IconData icon, int index, bool isDarkMode) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        IconButton(
          icon: Icon(
            icon,
            color: _selectedIndex == index
                ? const Color.fromARGB(255, 255, 87, 87)
                : isDarkMode
                    ? Color.fromARGB(255, 245, 242, 235)
                    : Colors.black54,
          ),
          onPressed: () => _onItemTapped(index),
        ),
        if (_selectedIndex == index)
          Container(
            height: 3,
            width: 20,
            color: const Color.fromARGB(255, 255, 87, 87),
          ),
      ],
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
