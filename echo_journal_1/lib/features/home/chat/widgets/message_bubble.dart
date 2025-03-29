import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:echo_journal_1/data/models/direct_message.dart';
import 'package:echo_journal_1/services/chat/chat_service.dart';

class MessageBubble extends StatefulWidget {
  final DirectMessage message;
  final bool isMe;
  final bool isDarkMode;

  const MessageBubble({
    super.key,
    required this.message,
    required this.isMe,
    required this.isDarkMode,
  });

  @override
  State<MessageBubble> createState() => _MessageBubbleState();
}

class _MessageBubbleState extends State<MessageBubble> {
  final ChatService _chatService = ChatService();
  bool _isMarking = false;

  Future<void> _markAsRead() async {
    if (widget.isMe || _isMarking || widget.message.isRead) return;

    setState(() => _isMarking = true);
    try {
      await _chatService.markMessagesAsRead(widget.message.sender);
    } catch (e) {
      debugPrint('❌ Error marking message as read: $e');
    } finally {
      if (mounted) {
        setState(() => _isMarking = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final timeString = DateFormat('h:mm a').format(widget.message.timestamp);
    final bubbleColor = widget.isMe
        ? Theme.of(context).primaryColor
        : widget.isDarkMode
            ? Colors.grey[800]
            : Colors.grey[200];
    final textColor = widget.isMe
        ? Colors.white
        : widget.isDarkMode
            ? Colors.white
            : Colors.black;
    final alignment =
        widget.isMe ? CrossAxisAlignment.end : CrossAxisAlignment.start;
    final bubbleRadius = BorderRadius.only(
      topLeft: Radius.circular(widget.isMe ? 16 : 4),
      topRight: Radius.circular(widget.isMe ? 4 : 16),
      bottomLeft: Radius.circular(16),
      bottomRight: Radius.circular(16),
    );

    return GestureDetector(
      onTap: _markAsRead,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Column(
          crossAxisAlignment: alignment,
          children: [
            Container(
              decoration: BoxDecoration(
                color: bubbleColor,
                borderRadius: bubbleRadius,
              ),
              padding: EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    widget.message.content,
                    style: TextStyle(color: textColor),
                  ),
                  SizedBox(height: 4),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        timeString,
                        style: TextStyle(
                          color: textColor.withOpacity(0.7),
                          fontSize: 10,
                        ),
                      ),
                      if (!widget.isMe) ...[
                        SizedBox(width: 4),
                        Icon(
                          widget.message.isRead ? Icons.done_all : Icons.done,
                          size: 12,
                          color: textColor.withOpacity(0.7),
                        ),
                      ],
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
