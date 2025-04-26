import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import 'package:echo_journal1/services/journal/journal_service.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'package:echo_journal1/features/widgets/navbar.dart';
import 'package:echo_journal1/core/configs/api_config.dart';
import 'package:echo_journal1/core/providers/subscription_provider.dart';

class AnalyticsPage extends StatefulWidget {
  const AnalyticsPage({Key? key}) : super(key: key);

  @override
  _AnalyticsPageState createState() => _AnalyticsPageState();
}

class _AnalyticsPageState extends State<AnalyticsPage>
    with SingleTickerProviderStateMixin {
  final JournalService _journalService = JournalService();
  List<Map<String, dynamic>> _sentimentData = [];
  bool _isLoading = true;
  String _error = '';
  TabController? _tabController;
  DateTime? _lastDataFetchTime;
  int _lastJournalCount = 0;

  final List<String> _timeFrames = ['Daily', 'Weekly', 'Monthly'];

  // Map to store sentiment values for charts (0=negative, 1=neutral, 2=positive)
  final Map<String, int> _sentimentValues = {
    'negative': 0,
    'neutral': 1,
    'positive': 2,
  };

  // Cache for pre-calculated data
  final Map<String, List<Map<String, dynamic>>> _filteredDataCache = {};
  final Map<String, Map<String, int>> _distributionCache = {};
  final Map<String, List<FlSpot>> _lineChartDataCache = {};

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: _timeFrames.length, vsync: this);
    _initializeData();
  }

  @override
  void dispose() {
    _tabController?.dispose();
    super.dispose();
  }

  Future<void> _initializeData() async {
    if (!mounted) return;

    // Get subscription provider
    final subscriptionProvider =
        Provider.of<SubscriptionProvider>(context, listen: false);

    // Check if we have cached data
    final prefs = await SharedPreferences.getInstance();
    final String? cachedDataJson = prefs.getString('sentiment_data_cache');

    if (cachedDataJson != null) {
      try {
        final List<dynamic> jsonList = jsonDecode(cachedDataJson);
        setState(() {
          _sentimentData = jsonList.cast<Map<String, dynamic>>();
          _isLoading = false;
        });
      } catch (e) {
        // If cache is corrupted, load fresh data
        await _loadSentimentData();
      }
    } else {
      // No cached data, load fresh
      await _loadSentimentData();
    }
  }

  Future<void> _loadSentimentData() async {
    if (!mounted) return;

    setState(() {
      _isLoading = true;
      _error = '';
    });

    try {
      final response = await _journalService.analyzeAllSentiments();

      if (!mounted) return;

      // Filter out deleted journals and ensure all required fields are present
      final validData = response
          .where(
            (item) =>
                item != null &&
                item['journal_id'] != null &&
                item['title'] != null &&
                item['date'] != null &&
                item['sentiment'] != null &&
                item['is_deleted'] != true,
          )
          .toList();

      // Save to cache
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('sentiment_data_cache', jsonEncode(validData));
      await prefs.setInt('sentiment_data_last_fetch_time',
          DateTime.now().millisecondsSinceEpoch);

      setState(() {
        _sentimentData = validData;
        _isLoading = false;
        _lastDataFetchTime = DateTime.now();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<bool> _shouldFetchFreshData() async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Check if we have cached data
      final String? cachedDataJson = prefs.getString('sentiment_data_cache');
      if (cachedDataJson == null) {
        return true; // No cached data, should fetch
      }

      // Check when the data was last fetched
      final lastFetchTimeMs =
          prefs.getInt('sentiment_data_last_fetch_time') ?? 0;
      final lastFetchTime = DateTime.fromMillisecondsSinceEpoch(
        lastFetchTimeMs,
      );
      _lastDataFetchTime = lastFetchTime;

      // Check if it's been more than 30 minutes since last fetch
      final now = DateTime.now();
      final thirtyMinutesAgo = now.subtract(const Duration(minutes: 30));

      // Get current journal count to see if new journals were added
      final journalCount = await _getCurrentJournalCount();
      final cachedJournalCount =
          prefs.getInt('sentiment_data_journal_count') ?? 0;
      _lastJournalCount = cachedJournalCount;

      // Decide whether to fetch fresh data
      final shouldFetch = lastFetchTime.isBefore(thirtyMinutesAgo) ||
          journalCount != cachedJournalCount;

      if (!shouldFetch) {
        // Use cached data
        final List<dynamic> jsonList = jsonDecode(cachedDataJson);
        _sentimentData = jsonList.cast<Map<String, dynamic>>();
      }

      return shouldFetch;
    } catch (e) {
      print('Error checking if should fetch data: $e');
      return true; // In case of error, fetch fresh data
    }
  }

  Future<int> _getCurrentJournalCount() async {
    try {
      // Make a lightweight call to get just the count
      final response = await _journalService.getJournals(page: 1);
      final totalCount = response['total_count'] ?? 0;
      return totalCount;
    } catch (e) {
      print('Error getting journal count: $e');
      return 0;
    }
  }

  Future<void> _saveSentimentDataToCache(
    List<Map<String, dynamic>> data,
  ) async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Save the data
      final jsonString = jsonEncode(data);
      await prefs.setString('sentiment_data_cache', jsonString);

      // Save the timestamp
      final now = DateTime.now();
      await prefs.setInt(
        'sentiment_data_last_fetch_time',
        now.millisecondsSinceEpoch,
      );

      // Save journal count
      final journalCount = await _getCurrentJournalCount();
      await prefs.setInt('sentiment_data_journal_count', journalCount);

      _lastDataFetchTime = now;
      _lastJournalCount = journalCount;
    } catch (e) {
      print('Error saving sentiment data to cache: $e');
    }
  }

  void _clearCalculatedCaches() {
    _filteredDataCache.clear();
    _distributionCache.clear();
    _lineChartDataCache.clear();
  }

  // Filter data based on selected time frame
  List<Map<String, dynamic>> _getFilteredData(String timeFrame) {
    // Check cache first
    if (_filteredDataCache.containsKey(timeFrame)) {
      return _filteredDataCache[timeFrame]!;
    }

    if (_sentimentData.isEmpty) return [];

    // Sort all data by date first
    _sentimentData.sort((a, b) {
      final dateA = DateTime.parse(a['date']).toLocal();
      final dateB = DateTime.parse(b['date']).toLocal();
      return dateB.compareTo(dateA); // Most recent first
    });

    // Get the most recent date from the data
    final mostRecentDate =
        DateTime.parse(_sentimentData.first['date']).toLocal();
    DateTime cutoffDate;

    switch (timeFrame) {
      case 'Daily':
        // Get entries from the most recent day
        cutoffDate = DateTime(
          mostRecentDate.year,
          mostRecentDate.month,
          mostRecentDate.day,
        );
        break;
      case 'Weekly':
        // Last 7 days from most recent
        cutoffDate = mostRecentDate.subtract(const Duration(days: 7));
        break;
      case 'Monthly':
        // Last 30 days from most recent
        cutoffDate = mostRecentDate.subtract(const Duration(days: 30));
        break;
      default:
        _filteredDataCache[timeFrame] = _sentimentData;
        return _sentimentData;
    }

    // Debug print for date filtering
    print('Filtering for $timeFrame');
    print('Most recent date: $mostRecentDate');
    print('Cutoff date: $cutoffDate');

    // Filter entries based on date
    final filteredData = _sentimentData.where((item) {
      if (item['date'] == null) return false;

      DateTime journalDate;
      try {
        journalDate = DateTime.parse(item['date']).toLocal();

        if (timeFrame == 'Daily') {
          // For daily view, include entries from the same day as most recent
          return journalDate.year == cutoffDate.year &&
              journalDate.month == cutoffDate.month &&
              journalDate.day == cutoffDate.day;
        }

        return journalDate.isAfter(cutoffDate) ||
            journalDate.isAtSameMomentAs(cutoffDate);
      } catch (e) {
        print('Error parsing date in _getFilteredData: ${item['date']} - $e');
        return false;
      }
    }).toList();

    // Debug print filtered results
    print('Filtered data count: ${filteredData.length}');
    if (filteredData.isNotEmpty) {
      print('First entry date: ${filteredData.first['date']}');
      print('Last entry date: ${filteredData.last['date']}');
    }

    // Cache the result
    _filteredDataCache[timeFrame] = filteredData;

    return filteredData;
  }

  // Calculate sentiment distribution for pie chart
  Map<String, int> _calculateSentimentDistribution(
    List<Map<String, dynamic>> data,
    String timeFrame,
  ) {
    // Check cache first
    if (_distributionCache.containsKey(timeFrame)) {
      return _distributionCache[timeFrame]!;
    }

    final Map<String, int> distribution = {
      'positive': 0,
      'negative': 0,
      'neutral': 0,
    };

    for (var item in data) {
      final sentiment = (item['sentiment'] ?? '').toString().toLowerCase();
      if (distribution.containsKey(sentiment)) {
        distribution[sentiment] = (distribution[sentiment] ?? 0) + 1;
      }
    }

    // Debug print
    print('$timeFrame distribution: ${distribution.toString()}');

    // Cache the result
    _distributionCache[timeFrame] = distribution;

    return distribution;
  }

  // Generate line chart data points with improved date labeling
  List<FlSpot> _generateLineChartData(
    List<Map<String, dynamic>> data,
    String timeFrame,
  ) {
    if (_lineChartDataCache.containsKey(timeFrame)) {
      return _lineChartDataCache[timeFrame]!;
    }

    if (data.isEmpty) return [];

    // Group by hour or day based on timeframe
    Map<String, List<Map<String, dynamic>>> groupedData = {};

    if (timeFrame == 'Daily') {
      // Create 24 hour slots
      for (int i = 0; i < 24; i++) {
        final hour = i.toString().padLeft(2, '0');
        groupedData['$hour:00'] = [];
      }
    }

    // Group the data
    for (var item in data) {
      if (item['date'] == null) continue;

      DateTime date;
      try {
        date = DateTime.parse(item['date']).toLocal();
      } catch (e) {
        continue;
      }

      String key;
      switch (timeFrame) {
        case 'Daily':
          key = DateFormat('HH:00').format(date); // Just hour for daily view
          break;
        case 'Weekly':
          key = DateFormat('EEE').format(date);
          break;
        case 'Monthly':
          key = DateFormat('MMM d').format(date);
          break;
        default:
          key = DateFormat('HH:00').format(date);
      }

      if (!groupedData.containsKey(key)) {
        groupedData[key] = [];
      }
      groupedData[key]!.add(item);
    }

    // Calculate average sentiment for each group
    List<MapEntry<String, double>> averages = [];

    groupedData.forEach((dateKey, items) {
      if (items.isEmpty) {
        // For empty slots, add null to maintain timeline
        averages.add(MapEntry(dateKey, -1)); // -1 will indicate no data
      } else {
        double total = 0;
        int count = 0;

        for (var item in items) {
          final sentiment = (item['sentiment'] ?? '').toString().toLowerCase();
          if (_sentimentValues.containsKey(sentiment)) {
            total += _sentimentValues[sentiment]!;
            count++;
          }
        }

        if (count > 0) {
          averages.add(MapEntry(dateKey, total / count));
        } else {
          averages.add(MapEntry(dateKey, -1));
        }
      }
    });

    // Sort by time
    averages.sort((a, b) => a.key.compareTo(b.key));

    // Store date labels for chart display
    _chartDateLabels[timeFrame] = averages.map((e) => e.key).toList();

    // Create spots, skipping points with no data
    final List<FlSpot> spots = [];
    for (int i = 0; i < averages.length; i++) {
      if (averages[i].value >= 0) {
        spots.add(FlSpot(i.toDouble(), averages[i].value));
      }
    }

    // Cache the result
    _lineChartDataCache[timeFrame] = spots;

    return spots;
  }

  // Map to store date labels for each time frame
  final Map<String, List<String>> _chartDateLabels = {};

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;
    final themeColor = Theme.of(context).primaryColor;

    return Scaffold(
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.only(top: 8),
            child: Text(
              'Mood Analytics',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Container(
            height: 40,
            child: TabBar(
              controller: _tabController,
              tabs: const [
                Tab(
                  child: Center(
                    child: Text(
                      'Daily',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ),
                Tab(
                  child: Center(
                    child: Text(
                      'Weekly',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ),
                Tab(
                  child: Center(
                    child: Text(
                      'Monthly',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ),
              ],
              labelColor: Colors.purple,
              unselectedLabelColor: Colors.grey,
              indicatorColor: Colors.purple,
              indicatorWeight: 2,
              indicatorSize: TabBarIndicatorSize.label,
              padding: EdgeInsets.zero,
              labelPadding: EdgeInsets.zero,
            ),
          ),
          Divider(
            height: 1,
            thickness: 1,
            color: Colors.grey.withOpacity(0.2),
          ),
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _error.isNotEmpty
                    ? Center(
                        child: Text(
                          'Error: $_error',
                          style: TextStyle(color: Colors.red),
                        ),
                      )
                    : _sentimentData.isEmpty
                        ? const Center(
                            child: Text(
                              'No sentiment data available. Add journals to see analytics.',
                            ),
                          )
                        : TabBarView(
                            controller: _tabController,
                            children: _timeFrames
                                .map(
                                  (timeFrame) => _buildTimeFrameAnalytics(
                                    timeFrame,
                                    isDarkMode,
                                    themeColor,
                                  ),
                                )
                                .toList(),
                          ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          await _forceClearCache();
        },
        tooltip: 'Refresh Data',
        child: const Icon(Icons.refresh),
      ),
    );
  }

  Widget _buildTimeFrameAnalytics(
    String timeFrame,
    bool isDarkMode,
    Color themeColor,
  ) {
    final filteredData = _getFilteredData(timeFrame);

    if (filteredData.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.analytics_outlined,
              size: 64,
              color: isDarkMode ? Colors.grey[700] : Colors.grey[300],
            ),
            const SizedBox(height: 16),
            Text(
              'No mood data available',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: isDarkMode ? Colors.grey[400] : Colors.grey[600],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Add some journal entries to see your mood analysis',
              style: TextStyle(
                fontSize: 14,
                color: isDarkMode ? Colors.grey[500] : Colors.grey[600],
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _loadSentimentData,
              icon: const Icon(Icons.refresh),
              label: const Text('Refresh Data'),
              style: ElevatedButton.styleFrom(
                backgroundColor: themeColor,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
              ),
            ),
          ],
        ),
      );
    }

    final distribution =
        _calculateSentimentDistribution(filteredData, timeFrame);
    final lineChartData = _generateLineChartData(filteredData, timeFrame);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSummaryCard(filteredData, distribution, timeFrame, isDarkMode),
          const SizedBox(height: 24),
          _buildPieChart(distribution, isDarkMode),
          const SizedBox(height: 24),
          _buildLineChart(lineChartData, timeFrame, isDarkMode, themeColor),
          const SizedBox(height: 24),
          _buildRecentJournalsSection(filteredData, isDarkMode),
        ],
      ),
    );
  }

  Widget _buildSummaryCard(
    List<Map<String, dynamic>> data,
    Map<String, int> distribution,
    String timeFrame,
    bool isDarkMode,
  ) {
    // Determine dominant mood
    String dominantMood = 'neutral';
    int maxCount = 0;

    distribution.forEach((mood, count) {
      if (count > maxCount) {
        maxCount = count;
        dominantMood = mood;
      }
    });

    // Get mood icon and color
    IconData moodIcon;
    Color moodColor;

    switch (dominantMood) {
      case 'positive':
        moodIcon = Icons.sentiment_very_satisfied;
        moodColor = Colors.green;
        break;
      case 'negative':
        moodIcon = Icons.sentiment_very_dissatisfied;
        moodColor = Colors.red;
        break;
      default:
        moodIcon = Icons.sentiment_neutral;
        moodColor = Colors.orange;
    }

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Mood Summary - $timeFrame',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Icon(moodIcon, size: 40, color: moodColor),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Predominant Mood: ${dominantMood.toUpperCase()}',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: moodColor,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Based on ${data.length} journal entries',
                        style: TextStyle(
                          color:
                              isDarkMode ? Colors.grey[300] : Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPieChart(Map<String, int> distribution, bool isDarkMode) {
    // Calculate total and percentages
    final int total = distribution.values.fold(0, (sum, count) => sum + count);

    // Prepare chart sections
    final List<PieChartSectionData> sections = [];

    final Map<String, Color> sentimentColors = {
      'positive': Colors.green,
      'neutral': Colors.orange,
      'negative': Colors.red,
    };

    distribution.forEach((sentiment, count) {
      if (count > 0) {
        final double percentage = (count / total) * 100;
        sections.add(
          PieChartSectionData(
            color: sentimentColors[sentiment] ?? Colors.grey,
            value: percentage,
            title: '${percentage.toStringAsFixed(1)}%',
            radius: 80,
            titleStyle: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
        );
      }
    });

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Mood Distribution',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 24),
            AspectRatio(
              aspectRatio: 1.5,
              child: PieChart(
                PieChartData(
                  sections: sections,
                  centerSpaceRadius: 40,
                  sectionsSpace: 4,
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildLegendItem('Positive', Colors.green),
                const SizedBox(width: 24),
                _buildLegendItem('Neutral', Colors.orange),
                const SizedBox(width: 24),
                _buildLegendItem('Negative', Colors.red),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegendItem(String label, Color color) {
    return Row(
      children: [
        Container(width: 16, height: 16, color: color),
        const SizedBox(width: 4),
        Text(label),
      ],
    );
  }

  Widget _buildLineChart(
    List<FlSpot> spots,
    String timeFrame,
    bool isDarkMode,
    Color themeColor,
  ) {
    if (spots.isEmpty) {
      return const SizedBox.shrink();
    }

    // Get date labels for this time frame
    final dateLabels = _chartDateLabels[timeFrame] ?? [];

    // Prepare bottom titles based on timeframe
    Widget bottomTitleWidgets(double value, TitleMeta meta) {
      if (value % 1 != 0) return const SizedBox.shrink();

      int index = value.toInt();
      if (index >= dateLabels.length) return const SizedBox.shrink();

      String title = dateLabels[index];

      // For daily view, only show every 3 hours to prevent overcrowding
      if (timeFrame == 'Daily' && index % 3 != 0) {
        return const SizedBox.shrink();
      }

      return Padding(
        padding: const EdgeInsets.only(top: 8.0),
        child: Text(
          title,
          style: TextStyle(
            fontSize: 10,
            color: isDarkMode ? Colors.grey[300] : Colors.grey[600],
          ),
        ),
      );
    }

    // Prepare left titles showing sentiment levels
    Widget leftTitleWidgets(double value, TitleMeta meta) {
      String text;

      if (value == 0) {
        text = 'Negative';
      } else if (value == 1) {
        text = 'Neutral';
      } else if (value == 2) {
        text = 'Positive';
      } else {
        return const SizedBox.shrink();
      }

      return Text(text, style: const TextStyle(fontSize: 10));
    }

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Mood Trends',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 24),
            AspectRatio(
              aspectRatio: 1.5,
              child: LineChart(
                LineChartData(
                  lineBarsData: [
                    LineChartBarData(
                      spots: spots,
                      isCurved: true,
                      color: themeColor,
                      barWidth: 3,
                      isStrokeCapRound: true,
                      dotData: const FlDotData(show: true),
                      belowBarData: BarAreaData(
                        show: true,
                        color: themeColor.withOpacity(0.2),
                      ),
                    ),
                  ],
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: bottomTitleWidgets,
                        reservedSize: 30,
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: leftTitleWidgets,
                        reservedSize: 60,
                      ),
                    ),
                    topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                  ),
                  gridData: FlGridData(
                    drawHorizontalLine: true,
                    drawVerticalLine: false,
                    horizontalInterval: 1,
                    getDrawingHorizontalLine: (value) {
                      return FlLine(
                        color: isDarkMode ? Colors.grey[700] : Colors.grey[300],
                        strokeWidth: 1,
                      );
                    },
                  ),
                  borderData: FlBorderData(
                    show: true,
                    border: Border.all(
                      color: isDarkMode ? Colors.grey[700]! : Colors.grey[300]!,
                      width: 1,
                    ),
                  ),
                  minY: 0,
                  maxY: 2,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecentJournalsSection(
    List<Map<String, dynamic>> data,
    bool isDarkMode,
  ) {
    // Sort by date, most recent first
    final recentJournals = List<Map<String, dynamic>>.from(data)
      ..sort((a, b) {
        if (a['date'] == null || b['date'] == null) return 0;

        DateTime dateA, dateB;
        try {
          dateA = DateTime.parse(a['date']);
          dateB = DateTime.parse(b['date']);
          return dateB.compareTo(dateA);
        } catch (e) {
          print('Error sorting dates: ${a['date']} or ${b['date']} - $e');
          return 0;
        }
      });

    // Take only the 5 most recent entries
    final displayJournals = recentJournals.take(5).toList();

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Recent Journal Entries',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            displayJournals.isEmpty
                ? const Center(child: Text('No recent journals available'))
                : ListView.builder(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: displayJournals.length,
                    itemBuilder: (context, index) {
                      final journal = displayJournals[index];
                      final sentiment =
                          (journal['sentiment'] ?? '').toString().toLowerCase();

                      IconData sentimentIcon;
                      Color sentimentColor;

                      switch (sentiment) {
                        case 'positive':
                          sentimentIcon = Icons.sentiment_very_satisfied;
                          sentimentColor = Colors.green;
                          break;
                        case 'negative':
                          sentimentIcon = Icons.sentiment_very_dissatisfied;
                          sentimentColor = Colors.red;
                          break;
                        default:
                          sentimentIcon = Icons.sentiment_neutral;
                          sentimentColor = Colors.orange;
                      }

                      String formattedDate = '';
                      try {
                        final date = DateTime.parse(journal['date']);
                        formattedDate = DateFormat('MMM d, yyyy').format(date);
                      } catch (e) {
                        print('Error formatting date: ${journal['date']} - $e');
                        formattedDate = 'Unknown date';
                      }

                      return Card(
                        margin: const EdgeInsets.only(bottom: 12),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(vertical: 8.0),
                          child: ListTile(
                            title: Text(
                              journal['title'] ?? 'Untitled',
                              style:
                                  const TextStyle(fontWeight: FontWeight.bold),
                              overflow: TextOverflow.ellipsis,
                              maxLines: 1,
                            ),
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(formattedDate),
                                const SizedBox(height: 4),
                                Text(
                                  journal['content'] ?? '',
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: isDarkMode
                                        ? Colors.grey[300]
                                        : Colors.grey[600],
                                  ),
                                ),
                              ],
                            ),
                            isThreeLine: true,
                            leading: CircleAvatar(
                              backgroundColor: sentimentColor.withOpacity(0.2),
                              child: Icon(sentimentIcon, color: sentimentColor),
                            ),
                            trailing: const Icon(
                              Icons.arrow_forward_ios,
                              size: 16,
                            ),
                            onTap: () {
                              // Show journal detail in a popup dialog
                              try {
                                // Print journal data for debugging
                                print('Journal data: ${jsonEncode(journal)}');

                                // Get the journal ID - try different possible field names
                                final journalId = journal['id'] ??
                                    journal['journal_id'] ??
                                    journal['pk'];

                                if (journalId == null) {
                                  throw Exception('Journal ID not found');
                                }

                                // Show loading dialog first
                                showDialog(
                                  context: context,
                                  barrierDismissible: false,
                                  builder: (BuildContext context) {
                                    return Dialog(
                                      shape: RoundedRectangleBorder(
                                        borderRadius: BorderRadius.circular(16),
                                      ),
                                      child: Padding(
                                        padding: const EdgeInsets.all(20.0),
                                        child: Column(
                                          mainAxisSize: MainAxisSize.min,
                                          children: [
                                            CircularProgressIndicator(),
                                            SizedBox(height: 16),
                                            Text('Loading journal...'),
                                          ],
                                        ),
                                      ),
                                    );
                                  },
                                );

                                // Fetch journal details from API
                                _fetchJournalDetails(journalId)
                                    .then((journalDetails) {
                                  // Close loading dialog
                                  Navigator.of(context).pop();
                                  print(
                                    'Journal details: ${jsonEncode(journalDetails)}',
                                  );

                                  // Extract data from the nested structure
                                  final journalData =
                                      journalDetails['data'] ?? journalDetails;

                                  final title = journalData['title'] ??
                                      journal['title'] ??
                                      'Journal Entry';
                                  final content = journalData['content'] ??
                                      journalData['journal_content'] ??
                                      'No content available';
                                  final date =
                                      journalData['date'] ?? journal['date'];

                                  String formattedDate = '';
                                  try {
                                    formattedDate = DateFormat(
                                      'MMM d, yyyy',
                                    ).format(DateTime.parse(date));
                                  } catch (e) {
                                    formattedDate = date ?? 'Unknown date';
                                  }

                                  // Show journal detail dialog with updated content
                                  showDialog(
                                    context: context,
                                    builder: (BuildContext context) {
                                      return Dialog(
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(
                                            16,
                                          ),
                                        ),
                                        elevation: 0,
                                        backgroundColor: Colors.transparent,
                                        child: Container(
                                          padding: EdgeInsets.all(16),
                                          decoration: BoxDecoration(
                                            color: isDarkMode
                                                ? Colors.grey[900]
                                                : Colors.white,
                                            shape: BoxShape.rectangle,
                                            borderRadius:
                                                BorderRadius.circular(16),
                                            boxShadow: [
                                              BoxShadow(
                                                color: Colors.black26,
                                                blurRadius: 10.0,
                                                offset: const Offset(
                                                  0.0,
                                                  10.0,
                                                ),
                                              ),
                                            ],
                                          ),
                                          child: Column(
                                            mainAxisSize: MainAxisSize.min,
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              // Header with title and close button
                                              Row(
                                                mainAxisAlignment:
                                                    MainAxisAlignment
                                                        .spaceBetween,
                                                children: [
                                                  Expanded(
                                                    child: Text(
                                                      title,
                                                      style: TextStyle(
                                                        fontSize: 18,
                                                        fontWeight:
                                                            FontWeight.bold,
                                                      ),
                                                      overflow:
                                                          TextOverflow.ellipsis,
                                                    ),
                                                  ),
                                                  IconButton(
                                                    icon: Icon(Icons.close),
                                                    onPressed: () {
                                                      Navigator.of(
                                                        context,
                                                      ).pop();
                                                    },
                                                  ),
                                                ],
                                              ),

                                              // Date and sentiment
                                              Row(
                                                children: [
                                                  Icon(
                                                    sentiment == 'positive'
                                                        ? Icons
                                                            .sentiment_very_satisfied
                                                        : (sentiment ==
                                                                'negative'
                                                            ? Icons
                                                                .sentiment_very_dissatisfied
                                                            : Icons
                                                                .sentiment_neutral),
                                                    color: sentiment ==
                                                            'positive'
                                                        ? Colors.green
                                                        : (sentiment ==
                                                                'negative'
                                                            ? Colors.red
                                                            : Colors.orange),
                                                    size: 20,
                                                  ),
                                                  const SizedBox(width: 8),
                                                  Text(
                                                    sentiment.toUpperCase(),
                                                    style: TextStyle(
                                                      fontWeight:
                                                          FontWeight.bold,
                                                      color: sentiment ==
                                                              'positive'
                                                          ? Colors.green
                                                          : (sentiment ==
                                                                  'negative'
                                                              ? Colors.red
                                                              : Colors.orange),
                                                    ),
                                                  ),
                                                  Spacer(),
                                                  Text(
                                                    formattedDate,
                                                    style: TextStyle(
                                                      color: isDarkMode
                                                          ? Colors.grey[300]
                                                          : Colors.grey[600],
                                                      fontStyle:
                                                          FontStyle.italic,
                                                      fontSize: 12,
                                                    ),
                                                  ),
                                                ],
                                              ),

                                              Divider(height: 16),

                                              // Content in a scrollable container
                                              Container(
                                                constraints: BoxConstraints(
                                                  maxHeight: MediaQuery.of(
                                                        context,
                                                      ).size.height *
                                                      0.4,
                                                ),
                                                child: SingleChildScrollView(
                                                  child: Text(
                                                    content,
                                                    style: TextStyle(
                                                      fontSize: 14,
                                                      height: 1.5,
                                                    ),
                                                  ),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      );
                                    },
                                  );
                                }).catchError((error) {
                                  // Close loading dialog
                                  Navigator.of(context).pop();

                                  // Show error message
                                  ToastHelper.showError(
                                    context,
                                    'Could not load journal details: ${error.toString()}',
                                  );
                                });
                              } catch (e) {
                                print('Dialog error: $e');
                                ToastHelper.showError(
                                  context,
                                  'Could not display journal entry: $e',
                                );
                              }
                            },
                          ),
                        ),
                      );
                    },
                  ),
          ],
        ),
      ),
    );
  }

  Future<void> _forceClearCache() async {
    try {
      setState(() {
        _isLoading = true;
      });

      // Clear all in-memory caches
      _clearCalculatedCaches();
      _sentimentData = []; // Clear the current data

      // Clear SharedPreferences cache
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove('sentiment_data_cache');
      await prefs.remove('sentiment_data_last_fetch_time');
      await prefs.remove('sentiment_data_journal_count');

      // Set state to indicate fresh load
      _lastDataFetchTime = null;
      _lastJournalCount = 0;

      // Force fetch fresh data directly from the server
      final sentimentResults = await _journalService.analyzeAllSentiments();
      _sentimentData = sentimentResults;

      // Save to cache
      await _saveSentimentDataToCache(sentimentResults);

      setState(() {
        _isLoading = false;
      });

      // Show toast message
      if (mounted) {
        ToastHelper.showInfo(context, 'Cache cleared. Fresh data loaded.');
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
      print('Error clearing cache: $e');
      if (mounted) {
        ToastHelper.showError(context, 'Error refreshing data: $e');
      }
    }
  }

  // Add this method to fetch journal details from API
  Future<Map<String, dynamic>> _fetchJournalDetails(dynamic journalId) async {
    try {
      return await _journalService.getJournal(int.parse(journalId.toString()));
    } catch (e) {
      print('Error fetching journal details: $e');
      throw e;
    }
  }
}
