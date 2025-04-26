import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal1/providers/leaderboard_provider.dart';
import 'package:echo_journal1/widgets/circular_profile.dart';
import 'package:echo_journal1/models/user.dart';
import 'package:cached_network_image/cached_network_image.dart';

class LeaderboardPage extends StatefulWidget {
  const LeaderboardPage({Key? key}) : super(key: key);

  @override
  _LeaderboardPageState createState() => _LeaderboardPageState();
}

class _LeaderboardPageState extends State<LeaderboardPage> {
  @override
  void initState() {
    super.initState();
    // Fetch leaderboard data when page loads
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final provider = Provider.of<LeaderboardProvider>(context, listen: false);
      provider.fetchLeaderboard();
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Theme.of(context).brightness == Brightness.dark;
    return Consumer<LeaderboardProvider>(
      builder: (context, provider, child) {
        return Column(
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Leaderboard',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w600,
                      color: isDarkMode ? Colors.white : Colors.black87,
                    ),
                  ),
                  SizedBox(height: 16),
                ],
              ),
            ),
            Expanded(
              child: _buildLeaderboardList(provider, isDarkMode),
            ),
          ],
        );
      },
    );
  }

  Widget _buildLeaderboardList(LeaderboardProvider provider, bool isDarkMode) {
    if (provider.isLoading) {
      return Center(
        child: CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation<Color>(
            isDarkMode ? Colors.white : Colors.purple,
          ),
        ),
      );
    }

    if (provider.leaderboardData.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.emoji_events_outlined,
              size: 64,
              color:
                  (isDarkMode ? Colors.white : Colors.purple).withOpacity(0.7),
            ),
            SizedBox(height: 16),
            Text(
              'No data available',
              style: TextStyle(
                fontSize: 16,
                color: (isDarkMode ? Colors.white : Colors.purple)
                    .withOpacity(0.7),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: () => provider.refresh(),
      child: ListView.builder(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount:
            provider.leaderboardData.length + (provider.hasNextPage ? 1 : 0),
        itemBuilder: (context, index) {
          if (index == provider.leaderboardData.length) {
            if (provider.isLoading) {
              return Center(
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(
                      isDarkMode ? Colors.white : Colors.purple,
                    ),
                  ),
                ),
              );
            }
            return const SizedBox.shrink();
          }

          final item = provider.leaderboardData[index];
          final days = item['days'] ?? '0 days';
          final rank = item['rank'] ?? (index + 1);

          return Container(
            margin: EdgeInsets.only(bottom: 12),
            child: Row(
              children: [
                _buildUserAvatar(item, rank, isDarkMode),
                SizedBox(width: 12),
                Expanded(
                  child: Text(
                    item['username'] ?? 'Unknown',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: item['is_current_user'] == true
                          ? FontWeight.w600
                          : FontWeight.w500,
                      color: isDarkMode ? Colors.white : Colors.black87,
                    ),
                  ),
                ),
                Container(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color:
                        Color(0xFF9C27B0).withOpacity(isDarkMode ? 0.3 : 0.1),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    days,
                    style: TextStyle(
                      color: isDarkMode ? Colors.white : Colors.purple,
                      fontWeight: FontWeight.w500,
                      fontSize: 14,
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildUserAvatar(
      Map<String, dynamic> user, int rank, bool isDarkMode) {
    final bool isTopThree = rank <= 3;
    final double size = 48.0;
    final backgroundColor = isDarkMode
        ? Color(0xFF9C27B0).withOpacity(0.3)
        : Color(0xFF9C27B0).withOpacity(0.1);

    return Stack(
      children: [
        Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(
              color: backgroundColor,
              width: 2,
            ),
          ),
          child: ClipOval(
            child: user['profile_picture'] != null
                ? CachedNetworkImage(
                    imageUrl: user['profile_picture'],
                    fit: BoxFit.cover,
                    placeholder: (context, url) => Container(
                      color: backgroundColor,
                      child: Center(
                        child: CircularProgressIndicator(
                          valueColor: AlwaysStoppedAnimation<Color>(
                            isDarkMode ? Colors.white : Colors.purple,
                          ),
                        ),
                      ),
                    ),
                    errorWidget: (context, url, error) => Container(
                      color: backgroundColor,
                      child: Center(
                        child: Text(
                          user['username']?.substring(0, 1).toUpperCase() ??
                              '?',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w500,
                            color: isDarkMode ? Colors.white : Colors.purple,
                          ),
                        ),
                      ),
                    ),
                  )
                : Container(
                    color: backgroundColor,
                    child: Center(
                      child: Text(
                        user['username']?.substring(0, 1).toUpperCase() ?? '?',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w500,
                          color: isDarkMode ? Colors.white : Colors.purple,
                        ),
                      ),
                    ),
                  ),
          ),
        ),
        if (isTopThree)
          Positioned(
            right: 0,
            bottom: 0,
            child: Container(
              padding: EdgeInsets.all(2),
              decoration: BoxDecoration(
                color: Theme.of(context).scaffoldBackgroundColor,
                shape: BoxShape.circle,
                border: Border.all(
                  color: backgroundColor,
                  width: 2,
                ),
              ),
              child: Text(
                _getRankEmoji(rank),
                style: TextStyle(fontSize: 14),
              ),
            ),
          ),
        if (!isTopThree)
          Positioned(
            right: 0,
            bottom: 0,
            child: Container(
              width: 20,
              height: 20,
              decoration: BoxDecoration(
                color: Theme.of(context).scaffoldBackgroundColor,
                shape: BoxShape.circle,
                border: Border.all(
                  color: backgroundColor,
                  width: 2,
                ),
              ),
              child: Center(
                child: Text(
                  rank.toString(),
                  style: TextStyle(
                    color: isDarkMode ? Colors.white : Colors.purple,
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }

  String _getRankEmoji(int rank) {
    switch (rank) {
      case 1:
        return '🥇';
      case 2:
        return '🥈';
      case 3:
        return '🥉';
      default:
        return rank.toString();
    }
  }
}
