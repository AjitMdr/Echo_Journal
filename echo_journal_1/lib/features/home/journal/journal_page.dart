import 'package:flutter/material.dart';
import 'package:echo_journal_1/core/configs/theme/curved-pattern.dart';
import 'package:echo_journal_1/features/home/journal/widgets/journal_form.dart';
import 'package:echo_journal_1/features/home/journal/widgets/journal_list.dart';
import 'package:echo_journal_1/features/home/journal/widgets/journal_search.dart';
import 'package:echo_journal_1/features/home/journal/widgets/journal_selector.dart';
import 'package:echo_journal_1/services/journal/journal_service.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal_1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'package:intl/intl.dart';
import 'package:echo_journal_1/core/configs/api_config.dart';

class JournalPage extends StatefulWidget {
  const JournalPage({Key? key}) : super(key: key);

  @override
  _JournalPageState createState() => _JournalPageState();
}

class _JournalPageState extends State<JournalPage> {
  final JournalService _journalService = JournalService();
  final GlobalKey<JournalFormWidgetState> _formKey = GlobalKey();
  final ScrollController _scrollController = ScrollController();
  final TextEditingController _searchController = TextEditingController();
  List<Map<String, dynamic>> _journals = [];
  List<Map<String, dynamic>> _filteredJournals = [];
  bool _isLoading = false;
  String _selectedLanguage = 'en';
  bool _isFormVisible = true;
  String _currentSort = 'newest';
  Map<String, dynamic> _selectedJournal = {};
  bool _isEditing = false;
  final TextEditingController _titleController = TextEditingController();
  final TextEditingController _contentController = TextEditingController();

  // Simplified scroll state management - form visibility controlled manually
  bool _manuallyShowingForm = true; // Tracks if form was manually shown by user
  double _lastScrollPosition = 0.0; // Tracks the last scroll position

  @override
  void initState() {
    super.initState();
    _loadJournals();
    _setupScrollListener();
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _searchController.dispose();
    _titleController.dispose();
    _contentController.dispose();
    super.dispose();
  }

  void _setupScrollListener() {
    _scrollController.addListener(() {
      final currentOffset = _scrollController.offset;
      bool isScrollingUp = currentOffset < _lastScrollPosition;
      _lastScrollPosition = currentOffset;

      // Hide form when scrolling up or down past threshold
      if ((isScrollingUp || currentOffset > 20) && _isFormVisible) {
        setState(() {
          _isFormVisible = false;
          _manuallyShowingForm = false;
        });
      }
    });
  }

  void _handleSort(String sortBy) {
    setState(() {
      _currentSort = sortBy;
      switch (sortBy) {
        case 'newest':
          _filteredJournals.sort(
            (a, b) => DateTime.parse(
              b['created_at'] ?? b['date'],
            ).compareTo(DateTime.parse(a['created_at'] ?? a['date'])),
          );
          break;
        case 'oldest':
          _filteredJournals.sort(
            (a, b) => DateTime.parse(
              a['created_at'] ?? a['date'],
            ).compareTo(DateTime.parse(b['created_at'] ?? b['date'])),
          );
          break;
        case 'title':
          _filteredJournals.sort(
            (a, b) => (a['title'] ?? '').toString().toLowerCase().compareTo(
                  (b['title'] ?? '').toString().toLowerCase(),
                ),
          );
          break;
      }
    });
  }

  void _filterJournals(String query) {
    if (_journals.isEmpty) {
      print('No journals to filter'); // Debug log
      return;
    }

    setState(() {
      if (query.isEmpty) {
        // Filter out deleted journals first
        _filteredJournals = _journals
            .where((j) => j != null && j['is_deleted'] != true)
            .toList();
      } else {
        _filteredJournals = _journals.where((journal) {
          if (journal == null ||
              journal['is_deleted'] == true ||
              journal['title'] == null ||
              journal['content'] == null) {
            print('Invalid or deleted journal: $journal'); // Debug log
            return false;
          }
          final title = journal['title'].toString().toLowerCase();
          final content = journal['content'].toString().toLowerCase();
          final searchLower = query.toLowerCase();
          return title.contains(searchLower) || content.contains(searchLower);
        }).toList();
      }
      print('Filtered journals: $_filteredJournals'); // Debug log
      _handleSort(_currentSort); // Apply current sort to filtered results
    });
  }

  Future<void> _loadJournals() async {
    if (!mounted) return;

    setState(() => _isLoading = true);
    try {
      final response = await _journalService.getJournals();
      print('Journal response: $response'); // Debug log

      if (!mounted) return;

      if (response['data'] == null) {
        print('No data field in response'); // Debug log
        throw Exception('Invalid response format');
      }

      final journalList = List<Map<String, dynamic>>.from(
        response['data'],
      ).where((j) => j != null && j['is_deleted'] != true).toList();
      print('Parsed journals: $journalList'); // Debug log

      setState(() {
        _journals = journalList;
        _filteredJournals = List.from(_journals);
        _handleSort(_currentSort); // Apply current sort to loaded journals
      });

      print('Journals state: $_journals'); // Debug log
      print('Filtered journals state: $_filteredJournals'); // Debug log
    } catch (e) {
      print('Error loading journals: $e'); // Debug log
      if (!mounted) return;

      String errorMessage = e.toString().replaceAll('Exception: ', '');
      ToastHelper.showError(context, errorMessage);

      setState(() {
        _journals = [];
        _filteredJournals = [];
      });
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _handleSubmit(Map<String, dynamic> journal) async {
    try {
      setState(() => _isLoading = true);

      // Validate the journal data
      if (journal['title']?.trim().isEmpty ?? true) {
        throw Exception('Title is required');
      }
      if (journal['content']?.trim().isEmpty ?? true) {
        throw Exception('Content is required');
      }
      if (!['en', 'ne'].contains(journal['language'])) {
        throw Exception('Invalid language selection');
      }

      await _journalService.createJournal({
        'title': journal['title'].trim(),
        'content': journal['content'].trim(),
        'language': journal['language'],
      });

      setState(() {
        _isFormVisible = false;
        _manuallyShowingForm = false;
      });

      await _loadJournals();

      if (mounted) {
        ToastHelper.showSuccess(context, 'Journal created successfully');
      }
    } catch (e) {
      if (mounted) {
        String errorMessage = e.toString();
        // Clean up the error message
        errorMessage = errorMessage.replaceAll('Exception: ', '');
        ToastHelper.showError(context, errorMessage);
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _handleUpdate(Map<String, dynamic> journal, String id) async {
    try {
      setState(() => _isLoading = true);
      await _journalService.updateJournal(int.parse(id), {
        'title': journal['title'],
        'content': journal['content'],
        'language': journal['language'],
      });

      // Clear editing state
      setState(() {
        _selectedJournal = {};
        _isEditing = false;
        _isFormVisible = false;
        _manuallyShowingForm = false;
        _titleController.clear();
        _contentController.clear();
      });

      // Reload the journal list
      await _loadJournals();

      if (mounted) {
        ToastHelper.showSuccess(context, 'Journal updated successfully');
      }
    } catch (e) {
      if (mounted) {
        ToastHelper.showError(context, 'Failed to update journal: $e');
      }
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _handleEdit(Map<String, dynamic> journal) {
    // Set the form values from the selected journal
    _titleController.text = journal['title'] ?? '';
    _contentController.text = journal['content'] ?? '';

    // Set the editing state
    setState(() {
      _selectedJournal = journal;
      _isFormVisible = true;
      _isEditing = true;
      _manuallyShowingForm = true; // Mark as manually shown when editing
    });

    // Update the form widget state
    _formKey.currentState?.setEditingJournal(journal);

    // Scroll to top to show the edit form
    _scrollController.animateTo(
      0,
      duration: Duration(milliseconds: 300),
      curve: Curves.easeOut,
    );
  }

  void _handleDelete(Map<String, dynamic> journal) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Delete Journal'),
        content: Text('Are you sure you want to delete this journal?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              _deleteJournal(int.parse(journal['id'].toString()));
            },
            child: Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _deleteJournal(int journalId) async {
    setState(() {
      _isLoading = true;
    });

    try {
      await _journalService.deleteJournal(journalId);

      // Remove the journal from both lists
      setState(() {
        _journals.removeWhere((j) => j['id'] == journalId);
        _filteredJournals.removeWhere((j) => j['id'] == journalId);
      });

      if (mounted) {
        ToastHelper.showSuccess(context, 'Journal deleted successfully');
      }
    } catch (e) {
      if (mounted) {
        String errorMessage = e.toString().replaceAll('Exception: ', '');
        ToastHelper.showError(
          context,
          'Failed to delete journal: $errorMessage',
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _shareJournal(int journalId) async {
    try {
      // Since sharing functionality is removed, show a message
      ToastHelper.showInfo(context, 'Sharing functionality is not available');
    } catch (e) {
      ToastHelper.showError(context, 'Error: $e');
    }
  }

  // Future<void> _unshareJournal(int journalId) async {
  //   try {
  //     final response = await _journalService.unshareJournal(journalId);
  //     if (response) {
  //       ToastHelper.showSuccess(context, 'Journal unshared successfully');
  //       _loadJournals();
  //     } else {
  //       ToastHelper.showError(context, 'Failed to unshare journal');
  //     }
  //   } catch (e) {
  //     ToastHelper.showError(context, 'Error: $e');
  //   }
  // }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    bool isDarkMode = themeProvider.isDarkMode;

    return Scaffold(
      floatingActionButton: !_isFormVisible
          ? FloatingActionButton(
              onPressed: _showCreateForm,
              backgroundColor: const Color.fromARGB(255, 255, 87, 87),
              child: const Icon(Icons.add),
            )
          : null,
      body: Stack(
        children: [
          CustomPaint(
            painter: CurvedPatternPainter(),
            size: MediaQuery.of(context).size,
          ),
          CustomScrollView(
            controller: _scrollController,
            physics: ClampingScrollPhysics(),
            slivers: [
              // Create Journal Form - Collapsible
              SliverToBoxAdapter(
                child: AnimatedContainer(
                  duration: Duration(milliseconds: 250),
                  height: _isFormVisible
                      ? 420
                      : 0, // Increased height to prevent overflow
                  curve: Curves.easeInOut,
                  child: Container(
                    color: Theme.of(context).scaffoldBackgroundColor,
                    padding: EdgeInsets.symmetric(horizontal: 12.0),
                    child: AnimatedOpacity(
                      duration: Duration(milliseconds: 200),
                      opacity: _isFormVisible ? 1.0 : 0.0,
                      child: _isFormVisible
                          ? SingleChildScrollView(
                              child: JournalFormWidget(
                                key: _formKey,
                                onSubmit: _handleSubmit,
                                onUpdate: _handleUpdate,
                                isDarkMode: isDarkMode,
                              ),
                            )
                          : SizedBox(),
                    ),
                  ),
                ),
              ),

              // Journal list header and search
              _isLoading
                  ? SliverFillRemaining(
                      child: Center(child: CircularProgressIndicator()),
                    )
                  : SliverToBoxAdapter(
                      child: Column(
                        children: [
                          Container(
                            padding: const EdgeInsets.fromLTRB(
                              16.0,
                              24.0,
                              16.0,
                              12.0,
                            ),
                            color: Theme.of(context).scaffoldBackgroundColor,
                            child: Column(
                              children: [
                                Row(
                                  children: [
                                    Icon(
                                      Icons.book,
                                      size: 16,
                                      color: Colors.blueGrey,
                                    ),
                                    SizedBox(width: 8),
                                    Text(
                                      'My Journals',
                                      style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.blueGrey,
                                      ),
                                    ),
                                    SizedBox(width: 8),
                                    Expanded(
                                      child: Divider(
                                        thickness: 2,
                                        color: Colors.blueGrey.withOpacity(0.3),
                                      ),
                                    ),
                                    IconButton(
                                      icon: Icon(Icons.psychology, size: 20),
                                      tooltip: 'Analyze All Sentiments',
                                      onPressed: () async {
                                        try {
                                          final results = await _journalService
                                              .analyzeAllSentiments();
                                          setState(() {
                                            for (var result in results) {
                                              final journalId =
                                                  result['journal_id'];
                                              final index =
                                                  _filteredJournals.indexWhere(
                                                (j) =>
                                                    j['id'].toString() ==
                                                    journalId.toString(),
                                              );
                                              if (index != -1) {
                                                _filteredJournals[index]
                                                        ['sentiment'] =
                                                    result['sentiment'];
                                              }
                                            }
                                          });
                                          if (mounted) {
                                            ToastHelper.showSuccess(
                                              context,
                                              'Analyzed all journal sentiments',
                                            );
                                          }
                                        } catch (e) {
                                          if (mounted) {
                                            ToastHelper.showError(
                                              context,
                                              'Failed to analyze sentiments',
                                            );
                                          }
                                        }
                                      },
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                          // Search bar below My Journals title
                          if (!_isFormVisible)
                            Container(
                              color: Theme.of(context).scaffoldBackgroundColor,
                              padding: EdgeInsets.symmetric(
                                horizontal: 16.0,
                                vertical: 8.0,
                              ),
                              child: JournalSearchBar(
                                key: ValueKey('search'),
                                searchController: _searchController,
                                onSearch: _filterJournals,
                                isDarkMode: isDarkMode,
                                onSortChanged: _handleSort,
                                currentSort: _currentSort,
                              ),
                            ),
                        ],
                      ),
                    ),

              // Journal items
              if (_isLoading)
                SliverFillRemaining(
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text(
                          'Loading journals...',
                          style: TextStyle(color: Colors.grey, fontSize: 14),
                        ),
                      ],
                    ),
                  ),
                )
              else if (_filteredJournals.isEmpty)
                SliverFillRemaining(
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.book, size: 48, color: Colors.grey),
                        SizedBox(height: 12),
                        Text(
                          'No journals found',
                          style: TextStyle(fontSize: 16, color: Colors.grey),
                        ),
                        SizedBox(height: 4),
                        Text(
                          _isFormVisible
                              ? 'Create your first journal entry above'
                              : 'Tap + to create a new journal',
                          style: TextStyle(color: Colors.grey, fontSize: 13),
                        ),
                        SizedBox(height: 16),
                        if (!_isFormVisible)
                          ElevatedButton.icon(
                            onPressed: () {
                              print(
                                'Current journals: $_journals',
                              ); // Debug log
                              _loadJournals();
                            },
                            icon: Icon(Icons.refresh),
                            label: Text('Refresh'),
                          ),
                      ],
                    ),
                  ),
                )
              else
                SliverPadding(
                  padding: EdgeInsets.only(top: 8, bottom: 80),
                  sliver: SliverList(
                    delegate: SliverChildBuilderDelegate((context, index) {
                      if (index >= _filteredJournals.length) {
                        print(
                          'Invalid index: $index for length: ${_filteredJournals.length}',
                        ); // Debug log
                        return null;
                      }

                      final journal = _filteredJournals[index];
                      print(
                        'Building journal card for index $index: $journal',
                      ); // Debug log

                      DateTime? createdDate;
                      try {
                        createdDate = DateTime.parse(
                          journal['created_at'] ??
                              journal['date'] ??
                              DateTime.now().toIso8601String(),
                        );
                      } catch (e) {
                        print(
                          'Error parsing date for journal: $e',
                        ); // Debug log
                        createdDate = DateTime.now();
                      }

                      final String formattedDate = DateFormat(
                        'MMM dd, yyyy – HH:mm',
                      ).format(createdDate.toLocal());

                      return Padding(
                        padding: EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 4,
                        ),
                        child: _buildJournalCard(
                          context,
                          journal,
                          formattedDate,
                        ),
                      );
                    }, childCount: _filteredJournals.length),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildJournalCard(
    BuildContext context,
    Map<String, dynamic> journal,
    String formattedDate,
  ) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;
    final sentiment = journal['sentiment'];

    Icon getSentimentIcon() {
      if (sentiment == null)
        return Icon(Icons.mood, color: Colors.grey, size: 20);

      switch (sentiment.toString().toLowerCase()) {
        case 'positive':
          return Icon(
            Icons.sentiment_very_satisfied,
            color: Colors.green,
            size: 20,
          );
        case 'negative':
          return Icon(
            Icons.sentiment_very_dissatisfied,
            color: Colors.red,
            size: 20,
          );
        case 'neutral':
          return Icon(Icons.sentiment_neutral, color: Colors.orange, size: 20);
        default:
          return Icon(Icons.mood, color: Colors.grey, size: 20);
      }
    }

    return Card(
      color: isDarkMode ? Colors.grey[800] : Colors.white,
      margin: EdgeInsets.only(bottom: 4),
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: InkWell(
        onTap: () => _showJournalDetails(context, journal),
        child: Padding(
          padding: EdgeInsets.all(8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Row(
                      children: [
                        Expanded(
                          child: Text(
                            journal['title'] ?? 'No Title',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: isDarkMode ? Colors.white : Colors.black,
                            ),
                          ),
                        ),
                        SizedBox(width: 8),
                        InkWell(
                          onTap: () async {
                            try {
                              final result =
                                  await _journalService.analyzeSentiment(
                                int.parse(journal['id'].toString()),
                              );
                              setState(() {
                                journal['sentiment'] = result['sentiment'];
                              });
                            } catch (e) {
                              if (mounted) {
                                ToastHelper.showError(
                                  context,
                                  'Failed to analyze sentiment',
                                );
                              }
                            }
                          },
                          child: getSentimentIcon(),
                        ),
                      ],
                    ),
                  ),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: Icon(Icons.edit, color: Colors.blue, size: 20),
                        onPressed: () => _handleEdit(journal),
                        constraints: BoxConstraints(),
                        padding: EdgeInsets.all(8),
                      ),
                      IconButton(
                        icon: Icon(Icons.delete, color: Colors.red, size: 20),
                        onPressed: () => _handleDelete(journal),
                        constraints: BoxConstraints(),
                        padding: EdgeInsets.all(8),
                      ),
                    ],
                  ),
                ],
              ),
              SizedBox(height: 8),
              Text(
                journal['content'] ?? 'No Content',
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 14,
                  color: isDarkMode ? Colors.white70 : Colors.black87,
                ),
              ),
              SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    formattedDate,
                    style: TextStyle(
                      fontSize: 12,
                      fontStyle: FontStyle.italic,
                      color: isDarkMode ? Colors.white54 : Colors.black54,
                    ),
                  ),
                  if (sentiment != null)
                    Text(
                      sentiment.toString().toUpperCase(),
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: sentiment.toString().toLowerCase() == 'positive'
                            ? Colors.green
                            : sentiment.toString().toLowerCase() == 'negative'
                                ? Colors.red
                                : Colors.orange,
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

  void _showJournalDetails(BuildContext context, Map<String, dynamic> journal) {
    final createdDate = DateTime.parse(
      journal['created_at'] ?? journal['date'],
    );
    final formattedDate = DateFormat(
      'MMMM dd, yyyy – HH:mm',
    ).format(createdDate.toLocal());
    final isDarkMode =
        Provider.of<ThemeProvider>(context, listen: false).isDarkMode;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.8,
        decoration: BoxDecoration(
          color: isDarkMode ? Colors.grey[850] : Colors.white,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(20),
            topRight: Radius.circular(20),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: Text(
                          journal['title'] ?? 'No Title',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: isDarkMode ? Colors.white : Colors.black,
                          ),
                        ),
                      ),
                      IconButton(
                        icon: Icon(Icons.close),
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                    ],
                  ),
                  SizedBox(height: 8),
                  Text(
                    formattedDate,
                    style: TextStyle(
                      fontSize: 14,
                      fontStyle: FontStyle.italic,
                      color: isDarkMode ? Colors.white70 : Colors.black54,
                    ),
                  ),
                  Divider(height: 24),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Text(
                  journal['content'] ?? 'No Content',
                  style: TextStyle(
                    fontSize: 16,
                    height: 1.5,
                    color: isDarkMode ? Colors.white : Colors.black87,
                  ),
                ),
              ),
            ),
            Padding(
              padding: EdgeInsets.all(16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  ElevatedButton.icon(
                    icon: Icon(Icons.edit),
                    label: Text('Edit'),
                    onPressed: () {
                      Navigator.of(context).pop();
                      _handleEdit(journal);
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Add method to explicitly show form
  void _showCreateForm() {
    setState(() {
      _isFormVisible = true;
      _manuallyShowingForm = true;
      // Clear search when form becomes visible
      _searchController.clear();
      _filterJournals('');
      // Scroll to top
      _scrollController.animateTo(
        0,
        duration: Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });
  }
}

// Custom delegate for SliverPersistentHeader
class _SliverJournalFormDelegate extends SliverPersistentHeaderDelegate {
  final Widget child;
  final double minHeight;
  final double maxHeight;

  _SliverJournalFormDelegate({
    required this.child,
    required this.minHeight,
    required this.maxHeight,
  });

  @override
  Widget build(
    BuildContext context,
    double shrinkOffset,
    bool overlapsContent,
  ) {
    return child;
  }

  @override
  double get minExtent => minHeight;

  @override
  double get maxExtent => maxHeight;

  @override
  bool shouldRebuild(_SliverJournalFormDelegate oldDelegate) {
    return oldDelegate.minHeight != minHeight ||
        oldDelegate.maxHeight != maxHeight ||
        oldDelegate.child != child;
  }
}
