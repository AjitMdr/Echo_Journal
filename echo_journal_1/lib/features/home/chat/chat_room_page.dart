import 'package:echo_journal_1/features/home/chat/widgets/message_bubble.dart';
import 'package:echo_journal_1/services/chat/chat_service.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'package:echo_journal_1/data/models/direct_message.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ChatRoomPage extends StatefulWidget {
  final String roomId;
  final String roomName;
  final bool isDarkMode;

  const ChatRoomPage({
    super.key,
    required this.roomId,
    required this.roomName,
    required this.isDarkMode,
  });

  @override
  State<ChatRoomPage> createState() => _ChatRoomPageState();
}

class _ChatRoomPageState extends State<ChatRoomPage> {
  final ChatService _chatService = ChatService();
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<DirectMessage> _messages = [];
  bool _isLoading = false;
  String? _currentUsername;

  @override
  void initState() {
    super.initState();
    _getCurrentUsername();
    _loadChatHistory();
    _connectToChat();
    _setupMessageListener();
  }

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    _chatService.dispose();
    super.dispose();
  }

  Future<void> _getCurrentUsername() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _currentUsername = prefs.getString('username');
    });
  }

  Future<void> _loadChatHistory() async {
    setState(() => _isLoading = true);
    try {
      // This would need to be implemented in the ChatService
      // final messages = await _chatService.getChatRoomHistory(widget.roomId);
      // setState(() {
      //   _messages = messages;
      // });
      // _scrollToBottom();

      // For now, just set loading to false
      setState(() => _isLoading = false);
    } catch (e) {
      if (mounted) {
        ToastHelper.showError(context, 'Failed to load chat history: $e');
      }
      setState(() => _isLoading = false);
    }
  }

  Future<void> _connectToChat() async {
    try {
      // This would need to be implemented in the ChatService
      // await _chatService.connectToChatRoom(widget.roomId);
    } catch (e) {
      if (mounted) {
        ToastHelper.showError(context, 'Failed to connect to chat: $e');
      }
    }
  }

  void _setupMessageListener() {
    _chatService.messagesStream.listen(
      (message) {
        setState(() {
          _messages.add(message);
        });
        _scrollToBottom();
      },
      onError: (error) {
        if (mounted) {
          ToastHelper.showError(context, 'Error receiving message: $error');
        }
      },
    );
  }

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      Future.delayed(Duration(milliseconds: 100), () {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      });
    }
  }

  Future<void> _sendMessage() async {
    final message = _messageController.text.trim();
    if (message.isEmpty) return;

    try {
      // This would need to be implemented in the ChatService
      // await _chatService.sendChatRoomMessage(widget.roomId, message);
      _messageController.clear();
    } catch (e) {
      if (mounted) {
        ToastHelper.showError(context, 'Failed to send message: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final backgroundColor = widget.isDarkMode ? Colors.black : Colors.white;
    final textColor = widget.isDarkMode ? Colors.white : Colors.black;

    return Scaffold(
      backgroundColor: backgroundColor,
      appBar: AppBar(
        title: Text(widget.roomName),
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: Icon(Icons.refresh),
            onPressed: _loadChatHistory,
            tooltip: 'Refresh Messages',
          ),
        ],
      ),
      body: Column(
        children: [
          // Messages list
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _messages.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.chat_bubble_outline,
                              size: 64,
                              color: Colors.grey,
                            ),
                            SizedBox(height: 16),
                            Text(
                              'No messages yet',
                              style:
                                  TextStyle(fontSize: 16, color: Colors.grey),
                            ),
                            SizedBox(height: 8),
                            Text(
                              'Start the conversation!',
                              style:
                                  TextStyle(fontSize: 14, color: Colors.grey),
                            ),
                          ],
                        ),
                      )
                    : ListView.builder(
                        controller: _scrollController,
                        padding: EdgeInsets.all(16),
                        itemCount: _messages.length,
                        itemBuilder: (context, index) {
                          final message = _messages[index];
                          final isMe =
                              message.sender.toString() == _currentUsername;

                          return MessageBubble(
                            message: message,
                            isMe: isMe,
                            isDarkMode: widget.isDarkMode,
                          );
                        },
                      ),
          ),
          // Message input
          Container(
            padding: EdgeInsets.symmetric(horizontal: 8, vertical: 8),
            decoration: BoxDecoration(
              color: widget.isDarkMode ? Colors.grey[900] : Colors.grey[100],
              boxShadow: [
                BoxShadow(
                  color: Colors.black12,
                  blurRadius: 4,
                  offset: Offset(0, -2),
                ),
              ],
            ),
            child: SafeArea(
              top: false,
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _messageController,
                      style: TextStyle(color: textColor),
                      decoration: InputDecoration(
                        hintText: 'Type a message...',
                        hintStyle: TextStyle(color: Colors.grey),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24),
                          borderSide: BorderSide.none,
                        ),
                        filled: true,
                        fillColor:
                            widget.isDarkMode ? Colors.grey[800] : Colors.white,
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 10,
                        ),
                      ),
                      maxLines: null,
                      textCapitalization: TextCapitalization.sentences,
                      onSubmitted: (_) => _sendMessage(),
                    ),
                  ),
                  SizedBox(width: 8),
                  Material(
                    color: Theme.of(context).primaryColor,
                    borderRadius: BorderRadius.circular(24),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(24),
                      onTap: _sendMessage,
                      child: Container(
                        padding: EdgeInsets.all(10),
                        child: Icon(Icons.send, color: Colors.white),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
