import 'package:flutter/material.dart';
import 'package:echo_journal_1/services/streak/streak_service.dart';

class StreakBadgeWidget extends StatefulWidget {
  final int? userId;
  final bool showBadges;
  final bool isDarkMode;

  const StreakBadgeWidget({
    super.key,
    this.userId,
    this.showBadges = true,
    required this.isDarkMode,
  });

  @override
  State<StreakBadgeWidget> createState() => _StreakBadgeWidgetState();
}

class _StreakBadgeWidgetState extends State<StreakBadgeWidget> {
  final StreakService _streakService = StreakService();
  Map<String, dynamic>? _streakData;
  List<Map<String, dynamic>> _badges = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  @override
  void dispose() {
    // Clean up any resources if needed
    super.dispose();
  }

  Future<void> _loadData() async {
    if (!mounted) return;

    try {
      setState(() {
        _isLoading = true;
        _error = null;
      });

      // Load streak data
      if (widget.userId != null) {
        _streakData = await _streakService.getUserStreak(widget.userId!);
      } else {
        _streakData = await _streakService.getCurrentStreak();
      }

      if (!mounted) return;

      // Load badges if enabled
      if (widget.showBadges) {
        _badges = await _streakService.getUserBadges();
      }

      if (!mounted) return;

      setState(() {
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return Center(
        child: Text('Error: $_error', style: TextStyle(color: Colors.red)),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Streak Display
        if (_streakData != null)
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: widget.isDarkMode ? Colors.grey[800] : Colors.white,
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Current Streak',
                      style: TextStyle(
                        fontSize: 14,
                        color: widget.isDarkMode
                            ? Colors.grey[300]
                            : Colors.grey[600],
                      ),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        Text(
                          _streakData!['emoji']?.toString() ?? '💫',
                          style: const TextStyle(
                            fontSize: 24,
                            height: 1.0,
                            leadingDistribution: TextLeadingDistribution.even,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          '${_streakData!['current_streak'] ?? 0} days',
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      'Longest Streak',
                      style: TextStyle(
                        fontSize: 14,
                        color: widget.isDarkMode
                            ? Colors.grey[300]
                            : Colors.grey[600],
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${_streakData!['longest_streak'] ?? 0} days',
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

        // Badges Display
        if (widget.showBadges && _badges.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text(
            'Badges',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: widget.isDarkMode ? Colors.white : Colors.black,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _badges.map((badge) {
              return Tooltip(
                message: badge['badge']['description'],
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color:
                        widget.isDarkMode ? Colors.grey[700] : Colors.grey[200],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    badge['badge']['icon'],
                    style: const TextStyle(fontSize: 24),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ],
    );
  }
}
