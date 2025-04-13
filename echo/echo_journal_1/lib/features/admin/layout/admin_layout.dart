import 'package:flutter/material.dart';
import '../../../core/configs/theme/theme-provider.dart';
import 'package:provider/provider.dart';

class AdminLayout extends StatelessWidget {
  final Widget child;
  final String title;

  const AdminLayout({
    Key? key,
    required this.child,
    required this.title,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;
    final screenWidth = MediaQuery.of(context).size.width;
    final isWideScreen = screenWidth > 768;

    return Scaffold(
      backgroundColor: isDarkMode ? Colors.grey[900] : Colors.grey[100],
      appBar: AppBar(
        title: Text(
          title,
          style: TextStyle(
            color: isDarkMode ? Colors.white : Colors.black,
            fontWeight: FontWeight.bold,
          ),
        ),
        backgroundColor: isDarkMode ? Colors.grey[850] : Colors.white,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(
              isDarkMode ? Icons.light_mode : Icons.dark_mode,
              color: isDarkMode ? Colors.white : Colors.black,
            ),
            onPressed: () {
              final themeProvider = Provider.of<ThemeProvider>(
                context,
                listen: false,
              );
              themeProvider.toggleTheme();
            },
          ),
        ],
      ),
      body: Row(
        children: [
          if (isWideScreen) _buildSidebar(context, isDarkMode),
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(16),
              child: child,
            ),
          ),
        ],
      ),
      drawer: isWideScreen ? null : _buildDrawer(context, isDarkMode),
    );
  }

  Widget _buildNavItem(
    BuildContext context,
    IconData icon,
    String label,
    String route,
    bool isDarkMode,
  ) {
    final isCurrentRoute = ModalRoute.of(context)?.settings.name == route;

    return ListTile(
      leading: Icon(
        icon,
        color: isCurrentRoute
            ? Theme.of(context).primaryColor
            : (isDarkMode ? Colors.white70 : Colors.black54),
      ),
      title: Text(
        label,
        style: TextStyle(
          color: isCurrentRoute
              ? Theme.of(context).primaryColor
              : (isDarkMode ? Colors.white : Colors.black),
          fontWeight: isCurrentRoute ? FontWeight.bold : FontWeight.normal,
        ),
      ),
      onTap: () {
        if (!isCurrentRoute) {
          Navigator.pushReplacementNamed(context, route);
        }
      },
      tileColor: isCurrentRoute
          ? (isDarkMode ? Colors.grey[800] : Colors.grey[100])
          : null,
    );
  }

  Widget _buildSidebar(BuildContext context, bool isDarkMode) {
    return Container(
      width: 250,
      color: isDarkMode ? Colors.grey[850] : Colors.white,
      child: _buildNavItems(context, isDarkMode),
    );
  }

  Widget _buildDrawer(BuildContext context, bool isDarkMode) {
    return Drawer(
      child: Container(
        color: isDarkMode ? Colors.grey[850] : Colors.white,
        child: _buildNavItems(context, isDarkMode),
      ),
    );
  }

  Widget _buildNavItems(BuildContext context, bool isDarkMode) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Icon(
                Icons.admin_panel_settings,
                color: Theme.of(context).primaryColor,
                size: 32,
              ),
              const SizedBox(width: 16),
              Text(
                'Admin Panel',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: isDarkMode ? Colors.white : Colors.black,
                ),
              ),
            ],
          ),
        ),
        const Divider(),
        _buildNavItem(
          context,
          Icons.dashboard,
          'Dashboard',
          '/admin',
          isDarkMode,
        ),
        _buildNavItem(
          context,
          Icons.people,
          'Users',
          '/admin/users',
          isDarkMode,
        ),
        _buildNavItem(
          context,
          Icons.book,
          'Journals',
          '/admin/journals',
          isDarkMode,
        ),
        _buildNavItem(
          context,
          Icons.subscriptions,
          'Subscriptions',
          '/admin/subscriptions',
          isDarkMode,
        ),
        _buildNavItem(
          context,
          Icons.analytics,
          'Analytics',
          '/admin/analytics',
          isDarkMode,
        ),
        const Spacer(),
        const Divider(),
        ListTile(
          leading: Icon(
            Icons.exit_to_app,
            color: isDarkMode ? Colors.white70 : Colors.black54,
          ),
          title: Text(
            'Exit Admin',
            style: TextStyle(
              color: isDarkMode ? Colors.white : Colors.black,
            ),
          ),
          onTap: () {
            Navigator.pushNamedAndRemoveUntil(
              context,
              '/',
              (route) => false,
            );
          },
        ),
      ],
    );
  }
}
