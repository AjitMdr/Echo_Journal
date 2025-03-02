import 'package:flutter/material.dart';
import 'package:intl/intl.dart'; // Don't forget to add this package to your pubspec.yaml
import '../../../../services/journal/journal_service.dart';
import '../../../core/constants/curved_pattern.dart';
import 'package:provider/provider.dart';
import '../../../../themes/theme_provider.dart';

class MoodAnalysisPage extends StatefulWidget {
  @override
  _MoodAnalysisPageState createState() => _MoodAnalysisPageState();
}

class _MoodAnalysisPageState extends State<MoodAnalysisPage> {
  List<Map<String, dynamic>> _journals = [];
  bool _isLoading = true;
  Map<int, bool> _expanded = {}; // Track expanded state for each journal

  @override
  void initState() {
    super.initState();
    _fetchJournals();
  }

  Future<void> _fetchJournals() async {
    try {
      List<Map<String, dynamic>> journals =
          await JournalService.fetchJournals();
      setState(() {
        _journals = journals;
        _isLoading = false;
      });
    } catch (e) {
      print('❌ Error fetching journals: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  String _formatDateTime(String dateTime) {
    try {
      DateTime parsedDate = DateTime.parse(dateTime);
      return DateFormat('yyyy-MM-dd – hh:mm a').format(parsedDate);
    } catch (e) {
      return 'Unknown Date';
    }
  }

  // Function to show a dialog when the journal is tapped
  void _showAnalysisDialog(Map<String, dynamic> journal) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Choose for Analysis'),
          content: Text(
              'Do you want to choose this journal for mood analysis?\n\nTitle: ${journal['title']}'),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                print('Journal selected for analysis: ${journal['title']}');
              },
              child: Text('Choose'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return Scaffold(
      backgroundColor:
          isDarkMode ? Colors.grey[900] : Color.fromARGB(255, 245, 242, 235),
      body: Stack(
        children: [
          CustomPaint(
            painter: CurvedPatternPainter(),
            size: MediaQuery.of(context).size,
          ),
          Column(
            children: [
              Expanded(
                child: FractionallySizedBox(
                  alignment: Alignment.topCenter,
                  heightFactor: 0.7,
                  child: _isLoading
                      ? Center(child: CircularProgressIndicator())
                      : _journals.isEmpty
                          ? Center(
                              child: Text(
                                'No journal entries found.',
                                style: TextStyle(
                                  color:
                                      isDarkMode ? Colors.white : Colors.black,
                                  fontSize: 16,
                                ),
                              ),
                            )
                          : SingleChildScrollView(
                              child: Column(
                                children: List.generate(
                                  _journals.length,
                                  (index) {
                                    var journal = _journals[index];
                                    String formattedDate = journal['date'] !=
                                                null &&
                                            journal['date'].isNotEmpty
                                        ? DateFormat('yyyy-MM-dd – kk:mm')
                                            .format(
                                                DateTime.parse(journal['date'])
                                                    .toLocal())
                                        : 'Created: Unknown';

                                    return GestureDetector(
                                      onTap: () {
                                        setState(() {
                                          _expanded[index] = !(_expanded[
                                                  index] ??
                                              false); // Toggle the expanded state
                                        });
                                        _showAnalysisDialog(
                                            journal); // Show the popup on tap
                                      },
                                      child: Card(
                                        color: isDarkMode
                                            ? Colors.grey[800]
                                            : Colors.white,
                                        margin: EdgeInsets.symmetric(
                                            vertical: 6, horizontal: 8),
                                        elevation: 2,
                                        shape: RoundedRectangleBorder(
                                          borderRadius:
                                              BorderRadius.circular(8),
                                        ),
                                        child: ListTile(
                                          contentPadding: EdgeInsets.symmetric(
                                              vertical: 10, horizontal: 12),
                                          title: Text(
                                            journal['title'] ?? 'No Title',
                                            style: TextStyle(
                                              color: isDarkMode
                                                  ? Colors.white
                                                  : Colors.black,
                                              fontSize: 16,
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                          subtitle: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                journal['content'] ??
                                                    'No Content',
                                                style: TextStyle(
                                                  color: isDarkMode
                                                      ? Colors.grey[300]
                                                      : Colors.black87,
                                                  fontSize: 14,
                                                ),
                                              ),
                                              if (_expanded[index] ??
                                                  false) ...[
                                                SizedBox(height: 8),
                                                Text(
                                                  journal['content'] ??
                                                      'No Content',
                                                  style: TextStyle(
                                                    color: isDarkMode
                                                        ? Colors.grey[300]
                                                        : Colors.black87,
                                                    fontSize: 14,
                                                  ),
                                                ),
                                              ]
                                            ],
                                          ),
                                          trailing: Text(
                                            formattedDate,
                                            style: TextStyle(
                                              color: isDarkMode
                                                  ? Colors.grey[400]
                                                  : Colors.black54,
                                              fontSize: 12,
                                            ),
                                          ),
                                        ),
                                      ),
                                    );
                                  },
                                ),
                              ),
                            ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
