import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../themes/theme_provider.dart';
import 'profile.dart';
import '../../../services/auth/login_service.dart';
import '../authentication/login_page.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return AnimatedTheme(
      data: isDarkMode ? ThemeData.dark() : ThemeData.light(),
      curve: Curves.easeInOut,
      duration: const Duration(milliseconds: 5), // Faster transition
      child: Scaffold(
        body: SingleChildScrollView(
          child: Column(
            children: [
              const SizedBox(height: 20),
              // Account Section
              _buildSection(
                context,
                'Account',
                [
                  _buildSettingItem(
                    icon: Icons.person_outline,
                    title: 'Profile',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => ProfilePage(),
                        ),
                      );
                    },
                    isDarkMode: isDarkMode,
                  ),
                  _buildSettingItem(
                    icon: Icons.lock_outline,
                    title: 'Privacy',
                    onTap: () {},
                    isDarkMode: isDarkMode,
                  ),
                  _buildSettingItem(
                    icon: Icons.notifications_none,
                    title: 'Notifications',
                    onTap: () {},
                    isDarkMode: isDarkMode,
                  ),
                ],
              ),
              // App Settings Section
              _buildSection(
                context,
                'App Settings',
                [
                  _buildSettingItem(
                    icon: Icons.dark_mode_outlined,
                    title: 'Dark Mode',
                    trailing: Switch(
                      value: isDarkMode,
                      onChanged: (value) {
                        themeProvider.toggleTheme();
                      },
                      activeColor: const Color.fromARGB(255, 255, 87, 87),
                    ),
                    isDarkMode: isDarkMode,
                  ),
                ],
              ),
              // Support Section
              _buildSection(
                context,
                'Support',
                [
                  _buildSettingItem(
                    icon: Icons.help_outline,
                    title: 'Help Center',
                    onTap: () {},
                    isDarkMode: isDarkMode,
                  ),
                  _buildSettingItem(
                    icon: Icons.info_outline,
                    title: 'About',
                    onTap: () {},
                    isDarkMode: isDarkMode,
                  ),
                ],
              ),
              // Logout Button
              Padding(
                padding: const EdgeInsets.all(20),
                child: ElevatedButton(
                  onPressed: () async {
                    // Call the logout method from the login service
                    await AuthService.logout();

                    // Navigate to the login page
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                          builder: (context) => const LoginPage()),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color.fromARGB(255, 255, 87, 87),
                    minimumSize: const Size.fromHeight(50),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    elevation: 5,
                  ),
                  child: Text(
                    'Log Out',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: isDarkMode ? Colors.black : Colors.white,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSection(BuildContext context, String title, List<Widget> items) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          child: Text(
            title,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: isDarkMode ? Colors.white : Colors.black,
            ),
          ),
        ),
        Container(
          margin: const EdgeInsets.symmetric(horizontal: 20),
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: isDarkMode ? Colors.grey[850] : Colors.white,
            borderRadius: BorderRadius.circular(12),
            boxShadow: [
              BoxShadow(
                color: isDarkMode
                    ? Colors.black.withOpacity(0.5)
                    : Colors.grey.withOpacity(0.3),
                blurRadius: 6,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Column(children: items),
        ),
        const SizedBox(height: 20),
      ],
    );
  }

  Widget _buildSettingItem({
    required IconData icon,
    required String title,
    Widget? trailing,
    VoidCallback? onTap,
    required bool isDarkMode,
  }) {
    return ListTile(
      leading: Icon(
        icon,
        color: isDarkMode ? Colors.white : Colors.black54,
      ),
      title: Text(
        title,
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w500,
          color: isDarkMode ? Colors.white : Colors.black,
        ),
      ),
      trailing: trailing ??
          Icon(Icons.chevron_right,
              color: isDarkMode ? Colors.white : Colors.black54),
      onTap: onTap,
    );
  }
}
