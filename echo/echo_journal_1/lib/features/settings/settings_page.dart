import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/features/authentication/pages/login_page.dart';
import 'package:echo_journal1/services/auth/login_service.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../profile/profile_page.dart';
import 'account_settings_page.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({Key? key}) : super(key: key);

  @override
  _SettingsPageState createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  bool _isDarkMode = false;

  @override
  void initState() {
    super.initState();
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    _isDarkMode = themeProvider.isDarkMode;
  }

  Future<void> _handleLogout() async {
    try {
      await AuthService.logout();
      if (mounted) {
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (context) => const LoginPage()),
          (route) => false,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error logging out: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return Scaffold(
      backgroundColor: isDarkMode ? Colors.grey[900] : Colors.grey[50],
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildSettingsCard([
                _buildSettingsItem(
                  icon: Icons.person_outline,
                  title: 'Profile',
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const ProfilePage()),
                  ),
                  isDarkMode: isDarkMode,
                ),
                Divider(
                    height: 1,
                    color: isDarkMode ? Colors.grey[700] : Colors.grey[300]),
                _buildSettingsItem(
                  icon: Icons.manage_accounts,
                  title: 'Account',
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const AccountSettingsPage()),
                  ),
                  isDarkMode: isDarkMode,
                ),
                Divider(
                    height: 1,
                    color: isDarkMode ? Colors.grey[700] : Colors.grey[300]),
              ], isDarkMode),
              _buildSettingsCard([
                _buildSettingsItem(
                  icon: Icons.dark_mode_outlined,
                  title: 'Dark Mode',
                  trailing: Switch(
                    value: isDarkMode,
                    onChanged: (value) {
                      themeProvider.toggleTheme();
                    },
                  ),
                  isDarkMode: isDarkMode,
                ),
              ], isDarkMode),
              _buildSettingsCard([
                _buildSettingsItem(
                  icon: Icons.info_outline,
                  title: 'About',
                  onTap: () {
                    // Navigate to About page
                  },
                  isDarkMode: isDarkMode,
                ),
              ], isDarkMode),
              Padding(
                padding: const EdgeInsets.all(16),
                child: ElevatedButton(
                  onPressed: _handleLogout,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red[400],
                    foregroundColor: Colors.white,
                    minimumSize: const Size(double.infinity, 48),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: const Text(
                    'Log Out',
                    style: TextStyle(fontSize: 16),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSettingsCard(List<Widget> children, bool isDarkMode) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: isDarkMode ? Colors.grey[850] : Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(isDarkMode ? 0.3 : 0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: children,
      ),
    );
  }

  Widget _buildSettingsItem({
    required IconData icon,
    required String title,
    Widget? trailing,
    VoidCallback? onTap,
    required bool isDarkMode,
  }) {
    return ListTile(
      leading: Icon(
        icon,
        color: isDarkMode ? Colors.white70 : Colors.black87,
      ),
      title: Text(
        title,
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w500,
          color: isDarkMode ? Colors.white : Colors.black87,
        ),
      ),
      trailing: trailing ??
          Icon(
            Icons.chevron_right,
            color: isDarkMode ? Colors.white70 : Colors.black54,
          ),
      onTap: onTap,
    );
  }
}
