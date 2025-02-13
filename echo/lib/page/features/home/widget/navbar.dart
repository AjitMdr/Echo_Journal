import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class NavBar extends StatefulWidget {
  const NavBar({super.key});

  @override
  NavBarState createState() => NavBarState();
}

class NavBarState extends State<NavBar> {
  int _selectedIndex = 0;
  final PageController _pageController = PageController();

  // Dummy profile image URL (you can update this with actual data)
  String? profileImageUrl;

  void _onPageChanged(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  void _onItemTapped(int index) {
    _pageController.jumpToPage(index);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Container(
            color: Colors.white,
            padding: EdgeInsets.symmetric(vertical: 8),
            child: Column(
              children: [
                AppBar(
                  backgroundColor: Colors.white,
                  elevation: 0,
                  title: Text(
                    'echo',
                    style: TextStyle(
                      color: Color.fromARGB(255, 255, 87, 87),
                      fontWeight: FontWeight.bold,
                      fontSize: 24,
                    ),
                  ),
                  actions: [
                    IconButton(
                      icon: Icon(Icons.chat, color: Colors.black),
                      onPressed: () {},
                    ),
                    IconButton(
                      icon: Icon(Icons.search, color: Colors.black),
                      onPressed: () {},
                    ),
                    // Profile picture icon or default avatar
                    CircleAvatar(
                      backgroundImage: profileImageUrl != null
                          ? NetworkImage(profileImageUrl!)
                          : AssetImage('assets/default_avatar.png')
                              as ImageProvider,
                      radius: 18,
                    ),
                    SizedBox(width: 8),
                  ],
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildNavItem(Icons.home, 0),
                    _buildNavItem(Icons.people, 1),
                    _buildNavItem(
                        Icons.bar_chart, 2), // Replaced video icon with chart
                    _buildNavItem(FontAwesomeIcons.faceSmile,
                        3), // Replaced marketplace icon with emoji
                    _buildNavItem(Icons.notifications, 4),
                    _buildNavItem(Icons.menu, 5),
                  ],
                ),
              ],
            ),
          ),
          Expanded(
            child: PageView(
              controller: _pageController,
              onPageChanged: _onPageChanged,
              children: [
                Center(child: Text('Home Page')),
                Center(child: Text('Friends Page')),
                Center(child: Text('Mood Analysis')),
                Center(child: Text('Analytics')),
                Center(child: Text('Notifications Page')),
                Center(child: Text('Menu Page')),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNavItem(IconData icon, int index) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        IconButton(
          icon: Icon(icon,
              color: _selectedIndex == index
                  ? Color.fromARGB(255, 255, 87, 87)
                  : Colors.black54),
          onPressed: () => _onItemTapped(index),
        ),
        if (_selectedIndex == index)
          Container(
            height: 3,
            width: 20,
            color: Color.fromARGB(255, 255, 87, 87),
          ),
      ],
    );
  }
}
