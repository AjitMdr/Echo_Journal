import 'package:flutter/material.dart';
import 'package:echo_journal_1/services/journal/journal_service.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'package:echo_journal_1/constants/api_constants.dart';

class JournalDetailPage extends StatefulWidget {
  final int journalId;

  const JournalDetailPage({Key? key, required this.journalId})
      : super(key: key);

  @override
  _JournalDetailPageState createState() => _JournalDetailPageState();
}

class _JournalDetailPageState extends State<JournalDetailPage> {
  late JournalService _journalService;
  late String _title;
  late String _content;
  late String _sentiment;
  late DateTime _createdAt;
  late bool _isLoading;
  late Map<String, dynamic> _journal;

  @override
  void initState() {
    super.initState();
    _journalService = JournalService();
    _isLoading = true;
    _loadJournal();
  }

  Future<void> _loadJournal() async {
    if (!mounted) return;

    setState(() => _isLoading = true);

    try {
      final response = await _journalService.getJournal(widget.journalId);

      if (!mounted) return;

      if (response == null || response['data'] == null) {
        ToastHelper.showError(context, 'Journal not found');
        Navigator.pop(context);
        return;
      }

      final journalData = response['data'];

      // Validate journal data
      if (journalData['id'] == null ||
          journalData['title'] == null ||
          journalData['content'] == null) {
        throw Exception('Invalid journal data');
      }

      setState(() {
        _journal = journalData;
        _title = journalData['title'];
        _content = journalData['content'];
        _sentiment = journalData['sentiment'];
        _createdAt = DateTime.parse(journalData['created_at']);
      });
    } catch (e) {
      if (!mounted) return;
      String errorMessage = e.toString().replaceAll('Exception: ', '');
      ToastHelper.showError(context, 'Failed to load journal: $errorMessage');
      Navigator.pop(context);
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _deleteJournal() async {
    if (!mounted) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Journal'),
        content: const Text(
          'Are you sure you want to delete this journal? This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    setState(() => _isLoading = true);

    try {
      await _journalService.deleteJournal(widget.journalId);
      if (!mounted) return;

      ToastHelper.showSuccess(context, 'Journal deleted successfully');
      Navigator.pop(context, true); // Pass true to indicate successful deletion
    } catch (e) {
      if (!mounted) return;
      String errorMessage = e.toString().replaceAll('Exception: ', '');
      ToastHelper.showError(context, 'Failed to delete journal: $errorMessage');
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Implement the build method to return the UI components
    return Scaffold(
      appBar: AppBar(
        title: Text(_title),
        actions: [
          IconButton(icon: Icon(Icons.delete), onPressed: _deleteJournal),
        ],
      ),
      body: _isLoading
          ? CircularProgressIndicator()
          : Column(
              children: [
                Text(_title),
                Text(_content),
                Text(_sentiment),
                Text(_createdAt.toString()),
              ],
            ),
    );
  }
}
