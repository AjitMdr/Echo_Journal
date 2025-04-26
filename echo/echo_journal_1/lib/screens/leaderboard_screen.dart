import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal1/providers/leaderboard_provider.dart';
import 'package:echo_journal1/services/streak/streak_service.dart';

class LeaderboardScreen extends StatefulWidget {
  const LeaderboardScreen({Key? key}) : super(key: key);

  @override
  State<LeaderboardScreen> createState() => _LeaderboardScreenState();
}

class _LeaderboardScreenState extends State<LeaderboardScreen> {
  late final LeaderboardProvider _provider;

  @override
  void initState() {
    super.initState();
    debugPrint('🔄 LeaderboardScreen initState');
    _provider = LeaderboardProvider(StreakService());
    _fetchInitialData();
  }

  Future<void> _fetchInitialData() async {
    debugPrint('🔄 Fetching initial leaderboard data');
    await _provider.fetchLeaderboard();
  }

  @override
  Widget build(BuildContext context) {
    debugPrint('🔄 Building LeaderboardScreen');
    return ChangeNotifierProvider.value(
      value: _provider,
      child: const LeaderboardContent(),
    );
  }
}

class LeaderboardContent extends StatelessWidget {
  const LeaderboardContent({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    debugPrint('🔄 Building LeaderboardContent');
    return Scaffold(
      appBar: AppBar(
        title: const Text('Leaderboard'),
        centerTitle: true,
      ),
      body: Consumer<LeaderboardProvider>(
        builder: (context, provider, child) {
          if (provider.isLoading && provider.leaderboardData.isEmpty) {
            return const Center(child: CircularProgressIndicator());
          }

          if (provider.error != null && provider.leaderboardData.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    provider.error!,
                    style: const TextStyle(color: Colors.red),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () => provider.refresh(),
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          }

          return ListView.builder(
            itemCount: provider.leaderboardData.length + 1,
            itemBuilder: (context, index) {
              if (index == provider.leaderboardData.length) {
                if (provider.isLoading) {
                  return const Center(
                    child: Padding(
                      padding: EdgeInsets.all(16.0),
                      child: CircularProgressIndicator(),
                    ),
                  );
                }
                if (provider.currentPage < provider.totalPages) {
                  return Center(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: ElevatedButton(
                        onPressed: () => provider.loadNextPage(),
                        child: const Text('Load More'),
                      ),
                    ),
                  );
                }
                return const SizedBox.shrink();
              }

              final item = provider.leaderboardData[index];
              return LeaderboardItem(
                rank: item['rank'],
                username: item['username'] ?? 'Unknown',
                streakCount: item['streak_count']?.toString() ?? '0',
                badge: item['badge'],
                isCurrentUser: item['is_current_user'] ?? false,
              );
            },
          );
        },
      ),
    );
  }
}

class LeaderboardItem extends StatelessWidget {
  final int rank;
  final String username;
  final String streakCount;
  final String? badge;
  final bool isCurrentUser;

  const LeaderboardItem({
    Key? key,
    required this.rank,
    required this.username,
    required this.streakCount,
    this.badge,
    required this.isCurrentUser,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isCurrentUser
            ? Theme.of(context).primaryColor.withOpacity(0.1)
            : null,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          _buildRankWidget(context),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              username,
              style: TextStyle(
                fontWeight: isCurrentUser ? FontWeight.bold : FontWeight.normal,
                fontSize: 16,
              ),
            ),
          ),
          Text(
            '$streakCount days',
            style: TextStyle(
              color: Theme.of(context).primaryColor,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          if (badge != null) ...[
            const SizedBox(width: 8),
            Text(
              badge!,
              style: const TextStyle(fontSize: 20),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildRankWidget(BuildContext context) {
    final bool isTopThree = rank <= 3;
    final String rankDisplay =
        isTopThree ? _getRankEmoji(rank) : rank.toString();

    return Container(
      width: 32,
      height: 32,
      decoration: BoxDecoration(
        color:
            isTopThree ? null : Theme.of(context).primaryColor.withOpacity(0.1),
        shape: BoxShape.circle,
      ),
      child: Center(
        child: Text(
          rankDisplay,
          style: TextStyle(
            color: isTopThree ? null : Theme.of(context).primaryColor,
            fontWeight: FontWeight.bold,
            fontSize: isTopThree ? 20 : 16,
          ),
        ),
      ),
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
