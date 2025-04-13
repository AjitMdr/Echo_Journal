import 'package:flutter/material.dart';
import '../layout/admin_layout.dart';
import '../../../core/configs/theme/theme-provider.dart';
import 'package:provider/provider.dart';

class AdminDashboardPage extends StatefulWidget {
  const AdminDashboardPage({Key? key}) : super(key: key);

  @override
  _AdminDashboardPageState createState() => _AdminDashboardPageState();
}

class _AdminDashboardPageState extends State<AdminDashboardPage> {
  bool _isLoading = true;
  Map<String, dynamic> _stats = {};

  @override
  void initState() {
    super.initState();
    _loadDashboardStats();
  }

  Future<void> _loadDashboardStats() async {
    // TODO: Implement API call to get dashboard stats
    // For now using mock data
    await Future.delayed(const Duration(seconds: 1));
    if (mounted) {
      setState(() {
        _stats = {
          'totalUsers': 1250,
          'activeUsers': 890,
          'premiumUsers': 456,
          'totalJournals': 5678,
          'todayJournals': 123,
          'totalRevenue': 12567.89,
        };
        _isLoading = false;
      });
    }
  }

  Widget _buildStatCard(
    String title,
    String value,
    IconData icon,
    Color color,
    bool isDarkMode,
  ) {
    return Card(
      elevation: 2,
      color: isDarkMode ? Colors.grey[850] : Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: color, size: 24),
                const SizedBox(width: 12),
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 16,
                    color: isDarkMode ? Colors.white70 : Colors.black54,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              value,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: isDarkMode ? Colors.white : Colors.black,
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;

    return AdminLayout(
      title: 'Dashboard',
      child: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Overview',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: isDarkMode ? Colors.white : Colors.black,
                    ),
                  ),
                  const SizedBox(height: 24),
                  GridView.count(
                    crossAxisCount: MediaQuery.of(context).size.width > 1200
                        ? 3
                        : MediaQuery.of(context).size.width > 800
                            ? 2
                            : 1,
                    crossAxisSpacing: 16,
                    mainAxisSpacing: 16,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    children: [
                      _buildStatCard(
                        'Total Users',
                        _stats['totalUsers'].toString(),
                        Icons.people,
                        Colors.blue,
                        isDarkMode,
                      ),
                      _buildStatCard(
                        'Active Users',
                        _stats['activeUsers'].toString(),
                        Icons.person,
                        Colors.green,
                        isDarkMode,
                      ),
                      _buildStatCard(
                        'Premium Users',
                        _stats['premiumUsers'].toString(),
                        Icons.star,
                        Colors.amber,
                        isDarkMode,
                      ),
                      _buildStatCard(
                        'Total Journals',
                        _stats['totalJournals'].toString(),
                        Icons.book,
                        Colors.purple,
                        isDarkMode,
                      ),
                      _buildStatCard(
                        'Today\'s Journals',
                        _stats['todayJournals'].toString(),
                        Icons.edit,
                        Colors.orange,
                        isDarkMode,
                      ),
                      _buildStatCard(
                        'Total Revenue',
                        '\$${_stats['totalRevenue'].toStringAsFixed(2)}',
                        Icons.attach_money,
                        Colors.green,
                        isDarkMode,
                      ),
                    ],
                  ),
                  const SizedBox(height: 32),
                  // TODO: Add charts and graphs section
                ],
              ),
            ),
    );
  }
}
