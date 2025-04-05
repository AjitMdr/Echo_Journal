import 'package:flutter/material.dart';
import 'package:echo_fe/services/journal/journal_service.dart';
import 'package:echo_fe/core/configs/theme/theme-provider.dart';
import 'package:echo_fe/utils/toast_helper.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import 'package:echo_fe/core/configs/api_config.dart';
import 'package:echo_fe/core/providers/subscription_provider.dart';
import '../../../features/subscription/subscription_check_widget.dart';

class MoodPage extends StatefulWidget {
  const MoodPage({super.key});

  @override
  State<MoodPage> createState() => _MoodPageState();
}

class _MoodPageState extends State<MoodPage> {
  final JournalService _journalService = JournalService();
  List<Map<String, dynamic>> _journals = [];
  List<Map<String, dynamic>> _filteredJournals = [];
  bool _isLoading = false;
  bool _isLoadingMore = false;
  String _searchQuery = '';
  String _selectedLanguage = 'all';
  int _currentPage = 1;
  bool _hasMorePages = true;

  @override
  void initState() {
    super.initState();
    _loadJournals();
  }

  void _filterJournals() {
    setState(() {
      _filteredJournals = _journals.where((journal) {
        // First check if the journal is not deleted
        if (journal == null || journal['is_deleted'] == true) {
          return false;
        }

        final matchesSearch =
            journal['title'].toString().toLowerCase().contains(
                      _searchQuery.toLowerCase(),
                    ) ||
                journal['content'].toString().toLowerCase().contains(
                      _searchQuery.toLowerCase(),
                    );

        final matchesLanguage = _selectedLanguage == 'all' ||
            journal['language'] == _selectedLanguage;

        return matchesSearch && matchesLanguage;
      }).toList();
    });
  }

  Future<void> _loadJournals({bool loadMore = false}) async {
    if (!mounted) return;

    if (loadMore) {
      if (_isLoadingMore || !_hasMorePages) return;
      setState(() => _isLoadingMore = true);
    } else {
      setState(() => _isLoading = true);
      _currentPage = 1;
      _journals = [];
    }

    try {
      final response = await _journalService.getJournals(page: _currentPage);
      if (!mounted) return;

      if (response['data'] == null) {
        throw Exception('Invalid response format');
      }

      final newJournals = List<Map<String, dynamic>>.from(response['data']);

      // Filter out deleted and invalid journals
      final validJournals = newJournals
          .where(
            (j) =>
                j != null &&
                j['id'] != null &&
                j['title'] != null &&
                j['content'] != null &&
                j['is_deleted'] != true,
          )
          .toList();

      setState(() {
        if (loadMore) {
          _journals.addAll(validJournals);
        } else {
          _journals = validJournals;
        }
        _hasMorePages = validJournals.length >=
            10; // Assuming backend sends 10 items per page
        if (_hasMorePages) _currentPage++;
        _filterJournals();
      });
    } catch (e) {
      if (!mounted) return;
      String errorMessage = e.toString().replaceAll('Exception: ', '');
      ToastHelper.showError(context, 'Failed to load journals: $errorMessage');
      setState(() {
        if (!loadMore) {
          _journals = [];
          _filteredJournals = [];
        }
      });
    } finally {
      if (mounted) {
        setState(() {
          if (loadMore) {
            _isLoadingMore = false;
          } else {
            _isLoading = false;
          }
        });
      }
    }
  }

  Future<void> _analyzeSentiment(int journalId) async {
    // Check if user has premium access
    final subscriptionProvider =
        Provider.of<SubscriptionProvider>(context, listen: false);
    final isPremium = subscriptionProvider.subscription?.status == 'ACTIVE' &&
        subscriptionProvider.subscription?.planDetails?.planType == 'PREMIUM';

    if (!isPremium) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('Premium Feature'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.star,
                color: Colors.amber,
                size: 48,
              ),
              SizedBox(height: 16),
              Text(
                'Mood Analysis is a premium feature',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 8),
              Text(
                'Upgrade to premium to unlock AI-powered mood analysis and gain insights into your emotional well-being.',
                textAlign: TextAlign.center,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Maybe Later'),
              style: TextButton.styleFrom(
                foregroundColor: Colors.grey[600],
              ),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                Navigator.pushNamed(context, '/subscription/plans');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).primaryColor,
                foregroundColor: Colors.white,
                elevation: 2,
                padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
              child: Text('Upgrade Now'),
            ),
          ],
        ),
      );
      return;
    }

    try {
      final sentimentData = await _journalService.analyzeSentiment(journalId);
      if (!mounted) return;

      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('Sentiment Analysis'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Overall Sentiment:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 2),
              Row(
                children: [
                  Icon(
                    _getSentimentIcon(sentimentData['sentiment']),
                    color: _getSentimentColor(sentimentData['sentiment']),
                    size: 24,
                  ),
                  SizedBox(width: 8),
                  Text(
                    sentimentData['sentiment']?.toString().toUpperCase() ??
                        'UNKNOWN',
                    style: TextStyle(
                      color: _getSentimentColor(sentimentData['sentiment']),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              if (sentimentData['rule_based'] == true) ...[
                SizedBox(height: 12),
                Container(
                  padding: EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.amber[100],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.info_outline,
                        size: 16,
                        color: Colors.amber[900],
                      ),
                      SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Using rule-based analysis',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.amber[900],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              SizedBox(height: 16),
              Text(
                'Journal Title:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 4),
              Text(sentimentData['title'] ?? 'No Title'),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Close'),
            ),
          ],
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ToastHelper.showError(context, 'Failed to analyze sentiment: $e');
    }
  }

  Color _getSentimentColor(String? sentiment) {
    switch (sentiment?.toLowerCase()) {
      case 'positive':
        return Colors.green;
      case 'negative':
        return Colors.red;
      case 'neutral':
        return Colors.orange;
      default:
        return Colors.grey;
    }
  }

  IconData _getSentimentIcon(String? sentiment) {
    switch (sentiment?.toLowerCase()) {
      case 'positive':
        return Icons.sentiment_very_satisfied;
      case 'negative':
        return Icons.sentiment_very_dissatisfied;
      case 'neutral':
        return Icons.sentiment_neutral;
      default:
        return Icons.mood;
    }
  }

  void _showJournalDetails(Map<String, dynamic> journal) {
    final isDarkMode =
        Provider.of<ThemeProvider>(context, listen: false).isDarkMode;
    showDialog(
      context: context,
      builder: (context) => Dialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        child: Container(
          width: MediaQuery.of(context).size.width * 0.9,
          padding: EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      journal['title'] ?? 'No Title',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: isDarkMode ? Colors.white : Colors.black87,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: Icon(Icons.close),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              SizedBox(height: 16),
              Container(
                constraints: BoxConstraints(
                  maxHeight: MediaQuery.of(context).size.height * 0.5,
                ),
                child: SingleChildScrollView(
                  child: Text(
                    journal['content'] ?? 'No Content',
                    style: TextStyle(
                      fontSize: 16,
                      height: 1.5,
                      color: isDarkMode ? Colors.white70 : Colors.black87,
                    ),
                  ),
                ),
              ),
              SizedBox(height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  TextButton.icon(
                    onPressed: () {
                      Navigator.of(context).pop();
                      _analyzeSentiment(journal['id']);
                    },
                    icon: Icon(Icons.psychology),
                    label: Text('Analyze Sentiment'),
                  ),
                  Text(
                    DateFormat('MMM dd, yyyy – HH:mm').format(
                      DateTime.parse(
                        journal['created_at'] ?? journal['date'],
                      ).toLocal(),
                    ),
                    style: TextStyle(
                      fontSize: 12,
                      color: isDarkMode ? Colors.white54 : Colors.black54,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildJournalCard(Map<String, dynamic> journal) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;
    final createdDate = DateTime.parse(
      journal['created_at'] ?? journal['date'],
    );
    final formattedDate = DateFormat(
      'MMM dd, yyyy – HH:mm',
    ).format(createdDate.toLocal());

    return Hero(
      tag: 'journal_${journal['id']}',
      child: Card(
        margin: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        elevation: 2,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () => _showJournalDetails(journal),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Text(
                        journal['title'] ?? 'No Title',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: isDarkMode ? Colors.white : Colors.black87,
                        ),
                      ),
                    ),
                    Container(
                      decoration: BoxDecoration(
                        color: isDarkMode ? Colors.grey[700] : Colors.grey[200],
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: IconButton(
                        icon: Icon(
                          Icons.psychology,
                          color: Theme.of(context).primaryColor,
                        ),
                        onPressed: () => _analyzeSentiment(journal['id']),
                        tooltip: 'Analyze Sentiment',
                      ),
                    ),
                  ],
                ),
                SizedBox(height: 12),
                Text(
                  journal['content'] ?? 'No Content',
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 14,
                    height: 1.4,
                    color: isDarkMode ? Colors.white70 : Colors.black87,
                  ),
                ),
                SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.access_time,
                          size: 16,
                          color: isDarkMode ? Colors.white54 : Colors.black54,
                        ),
                        SizedBox(width: 4),
                        Text(
                          formattedDate,
                          style: TextStyle(
                            fontSize: 12,
                            color: isDarkMode ? Colors.white54 : Colors.black54,
                          ),
                        ),
                      ],
                    ),
                    Container(
                      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: isDarkMode ? Colors.grey[700] : Colors.grey[200],
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        journal['language'] == 'ne' ? 'नेपाली' : 'English',
                        style: TextStyle(
                          fontSize: 12,
                          color: isDarkMode ? Colors.white54 : Colors.black54,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;

    return Scaffold(
      backgroundColor: isDarkMode ? Color(0xFF121212) : Colors.grey[100],
      appBar: PreferredSize(
        preferredSize: Size.fromHeight(70),
        child: AppBar(
          backgroundColor: isDarkMode ? Color(0xFF1E1E1E) : Colors.white,
          elevation: 0,
          title: Text(
            'Journal Analysis',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 18,
              color: isDarkMode ? Colors.white : Colors.black87,
            ),
          ),
          centerTitle: true,
          bottom: PreferredSize(
            preferredSize: Size.fromHeight(40),
            child: Container(
              padding: EdgeInsets.fromLTRB(16, 0, 16, 4),
              child: Row(
                children: [
                  Expanded(
                    child: Container(
                      height: 36,
                      decoration: BoxDecoration(
                        color:
                            isDarkMode ? Color(0xFF2C2C2C) : Colors.grey[200],
                        borderRadius: BorderRadius.circular(18),
                      ),
                      child: TextField(
                        style: TextStyle(
                          color: isDarkMode ? Colors.white70 : Colors.black87,
                        ),
                        onChanged: (value) {
                          setState(() {
                            _searchQuery = value;
                            _filterJournals();
                          });
                        },
                        decoration: InputDecoration(
                          hintText: 'Search journals...',
                          hintStyle: TextStyle(
                            color: isDarkMode ? Colors.white38 : Colors.black38,
                          ),
                          prefixIcon: Icon(
                            Icons.search,
                            color: isDarkMode ? Colors.white54 : Colors.black54,
                          ),
                          border: InputBorder.none,
                          contentPadding: EdgeInsets.symmetric(horizontal: 16),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(width: 8),
                  Container(
                    height: 36,
                    padding: EdgeInsets.symmetric(horizontal: 8),
                    decoration: BoxDecoration(
                      color: isDarkMode ? Color(0xFF2C2C2C) : Colors.grey[200],
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: DropdownButtonHideUnderline(
                      child: DropdownButton<String>(
                        value: _selectedLanguage,
                        dropdownColor:
                            isDarkMode ? Color(0xFF2C2C2C) : Colors.grey[200],
                        style: TextStyle(
                          color: isDarkMode ? Colors.white70 : Colors.black87,
                          fontSize: 13,
                        ),
                        icon: Icon(
                          Icons.language,
                          size: 18,
                          color: isDarkMode ? Colors.white54 : Colors.black54,
                        ),
                        items: [
                          DropdownMenuItem(value: 'all', child: Text('All')),
                          DropdownMenuItem(value: 'en', child: Text('English')),
                          DropdownMenuItem(value: 'ne', child: Text('नेपाली')),
                        ],
                        onChanged: (value) {
                          setState(() {
                            _selectedLanguage = value!;
                            _filterJournals();
                          });
                        },
                      ),
                    ),
                  ),
                  SizedBox(width: 8),
                  Container(
                    height: 36,
                    width: 36,
                    decoration: BoxDecoration(
                      color: isDarkMode ? Color(0xFF2C2C2C) : Colors.grey[200],
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: IconButton(
                      icon: Icon(
                        Icons.refresh,
                        size: 18,
                        color: isDarkMode ? Colors.white54 : Colors.black54,
                      ),
                      onPressed: _loadJournals,
                      tooltip: 'Refresh Journals',
                      padding: EdgeInsets.zero,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(
                      Theme.of(context).primaryColor,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Loading journals...',
                    style: TextStyle(
                      color: isDarkMode ? Colors.white70 : Colors.black54,
                    ),
                  ),
                ],
              ),
            )
          : _filteredJournals.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.book_outlined,
                        size: 80,
                        color: isDarkMode ? Colors.white24 : Colors.black26,
                      ),
                      SizedBox(height: 16),
                      Text(
                        _journals.isEmpty
                            ? 'No journals found'
                            : 'No matching journals',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w500,
                          color: isDarkMode ? Colors.white70 : Colors.black87,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        _journals.isEmpty
                            ? 'Start writing to see your journals here'
                            : 'Try adjusting your search or filter',
                        style: TextStyle(
                          fontSize: 14,
                          color: isDarkMode ? Colors.white38 : Colors.black45,
                        ),
                      ),
                      SizedBox(height: 24),
                      if (_journals.isEmpty)
                        ElevatedButton.icon(
                          onPressed: _loadJournals,
                          icon: Icon(Icons.refresh),
                          label: Text('Refresh'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Theme.of(context).primaryColor,
                            foregroundColor: Colors.white,
                            padding: EdgeInsets.symmetric(
                              horizontal: 24,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                        ),
                    ],
                  ),
                )
              : RefreshIndicator(
                  onRefresh: () => _loadJournals(),
                  color: Theme.of(context).primaryColor,
                  child: ListView.builder(
                    key: ValueKey(_filteredJournals.length),
                    padding: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                    itemCount:
                        _filteredJournals.length + (_hasMorePages ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index == _filteredJournals.length) {
                        if (_isLoadingMore) {
                          return Padding(
                            padding: EdgeInsets.all(16),
                            child: Center(
                              child: CircularProgressIndicator(
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  Theme.of(context).primaryColor,
                                ),
                              ),
                            ),
                          );
                        }
                        return Padding(
                          padding: EdgeInsets.all(16),
                          child: Center(
                            child: ElevatedButton.icon(
                              onPressed: () => _loadJournals(loadMore: true),
                              icon: Icon(Icons.refresh),
                              label: Text('Load More'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Theme.of(context).primaryColor,
                                foregroundColor: Colors.white,
                                padding: EdgeInsets.symmetric(
                                  horizontal: 24,
                                  vertical: 12,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20),
                                ),
                              ),
                            ),
                          ),
                        );
                      }
                      return _buildJournalCard(_filteredJournals[index]);
                    },
                  ),
                ),
    );
  }
}
