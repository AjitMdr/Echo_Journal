import 'package:flutter/material.dart';

class JournalDialogUtils {
  static Future<void> showDeleteConfirmation(
    BuildContext context,
    Map<String, dynamic> journal,
    Function(int) onDelete,
  ) async {
    return showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Confirm Delete', style: TextStyle(fontSize: 16)),
          content: Text(
            'Are you sure you want to delete "${journal['title']}"?',
            style: TextStyle(fontSize: 14),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                onDelete(journal['id']);
              },
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              child: Text('Delete'),
            ),
          ],
        );
      },
    );
  }
}
