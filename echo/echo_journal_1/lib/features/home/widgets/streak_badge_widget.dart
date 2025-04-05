import 'package:flutter/material.dart';
import 'package:echo_fe/services/streak/streak_service.dart';

class StreakBadgeWidget extends StatefulWidget {
  const StreakBadgeWidget({super.key});

  @override
  State<StreakBadgeWidget> createState() => _StreakBadgeWidgetState();
}

class _StreakBadgeWidgetState extends State<StreakBadgeWidget> {
  final StreakService _streakService = StreakService();
  Map<String, dynamic>? _streakData;
  List<Map<String, dynamic>> _badges = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    try {
      final streakData = await _streakService.getCurrentStreak();
      final badges = await _streakService.getUserBadges();
      setState(() {
        _streakData = streakData;
        _badges = badges;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error loading streak data: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    return Column(
      children: [
        _buildStreakDisplay(),
        const SizedBox(height: 16),
        _buildBadgesGrid(),
      ],
    );
  }

  Widget _buildStreakDisplay() {
    if (_streakData == null) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primaryContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            _streakData!['emoji'] ?? '🔥',
            style: const TextStyle(fontSize: 24),
          ),
          const SizedBox(width: 8),
          Text(
            '${_streakData!['current_streak']} Day Streak',
            style: Theme.of(context).textTheme.titleLarge,
          ),
        ],
      ),
    );
  }

  Widget _buildBadgesGrid() {
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        crossAxisSpacing: 16,
        mainAxisSpacing: 16,
      ),
      itemCount: _badges.length,
      itemBuilder: (context, index) {
        final badge = _badges[index]['badge'];
        return Container(
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.secondaryContainer,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(badge['icon'] ?? '🏆', style: const TextStyle(fontSize: 32)),
              const SizedBox(height: 4),
              Text(
                badge['name'] ?? '',
                style: Theme.of(context).textTheme.bodySmall,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        );
      },
    );
  }
}
