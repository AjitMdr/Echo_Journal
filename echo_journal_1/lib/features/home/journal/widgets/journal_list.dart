import 'package:flutter/material.dart';
import 'package:intl/intl.dart' show DateFormat;
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';

class JournalListWidget extends StatelessWidget {
  final List<Map<String, dynamic>> journals;
  final bool isDarkMode;
  final Function(Map<String, dynamic>) showDeleteConfirmation;
  final Function(Map<String, dynamic>, String) onUpdate;
  final ScrollController scrollController;

  const JournalListWidget({
    super.key,
    required this.journals,
    required this.isDarkMode,
    required this.showDeleteConfirmation,
    required this.onUpdate,
    required this.scrollController,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.fromLTRB(16.0, 4.0, 16.0, 4.0),
          color: Theme.of(context).scaffoldBackgroundColor,
          child: Row(
            children: [
              Icon(Icons.book, size: 16, color: Colors.blueGrey),
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
            ],
          ),
        ),
        Expanded(
          child:
              journals.isEmpty
                  ? _buildEmptyListView()
                  : _buildJournalListView(context),
        ),
      ],
    );
  }

  Widget _buildEmptyListView() {
    return AnimationConfiguration.synchronized(
      duration: const Duration(milliseconds: 500),
      child: SlideAnimation(
        verticalOffset: 50.0,
        child: FadeInAnimation(
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
                  'Create your first journal entry above',
                  style: TextStyle(color: Colors.grey, fontSize: 13),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildJournalListView(BuildContext context) {
    return AnimationLimiter(
      child: ListView.builder(
        controller: scrollController,
        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        physics: const BouncingScrollPhysics(),
        itemCount: journals.length,
        itemBuilder: (context, index) {
          final journal = journals[index];
          final DateTime createdDate = DateTime.parse(
            journal['created_at'] ?? journal['date'],
          );
          final String formattedDate = DateFormat(
            'MMM dd, yyyy – HH:mm',
          ).format(createdDate.toLocal());

          return AnimationConfiguration.staggeredList(
            position: index,
            duration: const Duration(milliseconds: 375),
            child: SlideAnimation(
              verticalOffset: 50.0,
              child: FadeInAnimation(
                child: Dismissible(
                  key: Key(journal['id'].toString()),
                  background: Container(
                    color: Colors.red,
                    alignment: Alignment.centerRight,
                    padding: EdgeInsets.only(right: 20.0),
                    child: Icon(Icons.delete, color: Colors.white),
                  ),
                  direction: DismissDirection.endToStart,
                  confirmDismiss: (direction) async {
                    final confirmed = await showDialog(
                      context: context,
                      builder: (BuildContext context) {
                        return AlertDialog(
                          title: Text('Delete Journal'),
                          content: Text(
                            'Are you sure you want to delete this journal?',
                          ),
                          actions: <Widget>[
                            TextButton(
                              onPressed: () => Navigator.of(context).pop(false),
                              child: Text('Cancel'),
                            ),
                            TextButton(
                              onPressed: () => Navigator.of(context).pop(true),
                              child: Text(
                                'Delete',
                                style: TextStyle(color: Colors.red),
                              ),
                            ),
                          ],
                        );
                      },
                    );

                    if (confirmed == true) {
                      showDeleteConfirmation(journal);
                    }
                    return false;
                  },
                  child: Card(
                    color: isDarkMode ? Colors.grey[800] : Colors.white,
                    margin: EdgeInsets.only(bottom: 4),
                    elevation: 2,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
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
                                  child: Text(
                                    journal['title'] ?? 'No Title',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color:
                                          isDarkMode
                                              ? Colors.white
                                              : Colors.black,
                                    ),
                                  ),
                                ),
                                Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    IconButton(
                                      icon: Icon(
                                        Icons.edit,
                                        color: Colors.blue,
                                        size: 20,
                                      ),
                                      onPressed:
                                          () => onUpdate(
                                            journal,
                                            journal['id'].toString(),
                                          ),
                                      constraints: BoxConstraints(),
                                      padding: EdgeInsets.all(8),
                                    ),
                                    IconButton(
                                      icon: Icon(
                                        Icons.delete,
                                        color: Colors.red,
                                        size: 20,
                                      ),
                                      onPressed:
                                          () => showDeleteConfirmation(journal),
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
                                color:
                                    isDarkMode
                                        ? Colors.white70
                                        : Colors.black87,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              formattedDate,
                              style: TextStyle(
                                fontSize: 12,
                                fontStyle: FontStyle.italic,
                                color:
                                    isDarkMode
                                        ? Colors.white54
                                        : Colors.black54,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  void _showJournalDetails(BuildContext context, Map<String, dynamic> journal) {
    showDialog(
      context: context,
      builder:
          (context) => Dialog(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            child: Container(
              padding: EdgeInsets.all(16),
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
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      IconButton(
                        icon: Icon(Icons.close),
                        onPressed: () => Navigator.pop(context),
                      ),
                    ],
                  ),
                  Divider(),
                  SizedBox(height: 8),
                  Flexible(
                    child: SingleChildScrollView(
                      child: Text(
                        journal['content'] ?? 'No Content',
                        style: TextStyle(fontSize: 14),
                      ),
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Created: ${DateFormat('MMM dd, yyyy – HH:mm').format(DateTime.parse(journal['created_at'] ?? journal['date']).toLocal())}',
                    style: TextStyle(
                      fontSize: 12,
                      fontStyle: FontStyle.italic,
                      color: Colors.grey,
                    ),
                  ),
                ],
              ),
            ),
          ),
    );
  }
}
