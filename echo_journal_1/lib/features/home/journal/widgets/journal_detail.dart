import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class JournalDetail extends StatelessWidget {
  final Map<String, dynamic> journal;

  const JournalDetail({Key? key, required this.journal}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final sentiment = journal['sentiment'] ?? 'unknown';
    final createdAt =
        journal['created_at'] != null
            ? DateFormat(
              'MMM d, yyyy h:mm a',
            ).format(DateTime.parse(journal['created_at']))
            : 'Unknown date';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              journal['title'] ?? 'Untitled',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            IconButton(
              icon: const Icon(Icons.close),
              onPressed: () => Navigator.of(context).pop(),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Text(
          createdAt,
          style: Theme.of(
            context,
          ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Icon(
              sentiment == 'positive'
                  ? Icons.sentiment_satisfied
                  : sentiment == 'negative'
                  ? Icons.sentiment_dissatisfied
                  : Icons.sentiment_neutral,
              color:
                  sentiment == 'positive'
                      ? Colors.green
                      : sentiment == 'negative'
                      ? Colors.red
                      : Colors.grey,
              size: 28,
            ),
            const SizedBox(width: 8),
            Text(
              sentiment.toUpperCase(),
              style: TextStyle(
                color:
                    sentiment == 'positive'
                        ? Colors.green
                        : sentiment == 'negative'
                        ? Colors.red
                        : Colors.grey,
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        Text(
          'Language: ${journal['language']?.toUpperCase() ?? 'Unknown'}',
          style: const TextStyle(
            fontWeight: FontWeight.w500,
            color: Colors.grey,
          ),
        ),
        const SizedBox(height: 16),
        Expanded(
          child: SingleChildScrollView(
            child: Text(
              journal['content'] ?? '',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
          ),
        ),
      ],
    );
  }
}
